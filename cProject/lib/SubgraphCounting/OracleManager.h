#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

namespace GraphLib {
namespace CardinalityEstimation {

class OracleManager {
    // 存储 internal_id -> probability 的映射
    // 使用 vector 因为 internal_id 是从 0 到 N-1 的连续整数，访问速度最快
    std::vector<float> oracle_probs; 
    std::vector<double> agg_values;

    std::vector<std::string> parse_csv_line(const std::string& line) {
        std::vector<std::string> result;
        result.reserve(16);

        std::string cell;
        cell.reserve(64);

        bool inside_quotes = false;
        for (char c : line) {
            if (c == '"') {
                inside_quotes = !inside_quotes;
            } else if (c == ',' && !inside_quotes) {
                result.push_back(std::move(cell));
                cell.clear();
                cell.reserve(64);
            } else {
                cell.push_back(c);
            }
        }
        result.push_back(std::move(cell));
        return result;
    }

    void LoadDoublesFromCSV(const std::string& filepath,
                        const std::string& id_col_name,
                        const std::string& value_col_name,
                        const std::unordered_map<std::string, int>& id_map,
                        std::vector<double>& target,
                        int& updated_count) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "[Warning] Cannot open file: " << filepath << std::endl;
            return;
        }

        std::string line;
        int id_idx = -1;
        int val_idx = -1;

        if (std::getline(f, line)) {
            std::vector<std::string> header = parse_csv_line(line);
            for (size_t i = 0; i < header.size(); ++i) {
                std::string col = header[i];
                col.erase(0, col.find_first_not_of(" \t\r\n\""));
                col.erase(col.find_last_not_of(" \t\r\n\"") + 1);
                if (col == id_col_name) id_idx = (int)i;
                else if (col == value_col_name) val_idx = (int)i;
            }
        }

        if (id_idx == -1 || val_idx == -1) {
            std::cerr << "[Warning] Columns not found in " << filepath
                    << " (ID: " << id_col_name << ", Value: " << value_col_name << ")\n";
            return;
        }

        const int need_cols = std::max(id_idx, val_idx);
        while (std::getline(f, line)) {
            std::vector<std::string> row = parse_csv_line(line);
            if ((int)row.size() <= need_cols) continue;

            const std::string& orig_id = row[id_idx];
            auto it = id_map.find(orig_id);
            if (it == id_map.end()) continue;

            int internal_id = it->second;
            if (internal_id < 0 || internal_id >= (int)target.size()) continue;

            try {
                target[internal_id] = std::stod(row[val_idx]);
                ++updated_count;
            } catch (...) {
            }
        }
    }

    void LoadProbabilitiesFromCSV(const std::string& filepath,
                              const std::string& id_col_name,
                              const std::string& prob_col_name,
                              const std::unordered_map<std::string, int>& id_map,
                              int& updated_count) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "[Warning] Cannot open file: " << filepath << std::endl;
            return;
        }

        std::string line;
        int id_idx = -1;
        int prob_idx = -1;

        if (std::getline(f, line)) {
            std::vector<std::string> header = parse_csv_line(line);
            for (size_t i = 0; i < header.size(); ++i) {
                std::string col = header[i];
                col.erase(0, col.find_first_not_of(" \t\r\n\""));
                col.erase(col.find_last_not_of(" \t\r\n\"") + 1);
                if (col == id_col_name) id_idx = (int)i;
                else if (col == prob_col_name) prob_idx = (int)i;
            }
        }

        if (id_idx == -1 || prob_idx == -1) {
            std::cerr << "[Warning] Columns not found in " << filepath
                    << " (ID: " << id_col_name << ", Prob: " << prob_col_name << ")" << std::endl;
            return;
        }

        const int need_cols = std::max(id_idx, prob_idx);
        while (std::getline(f, line)) {
            std::vector<std::string> row = parse_csv_line(line);
            if ((int)row.size() <= need_cols) continue;

            const std::string& orig_id = row[id_idx];
            auto it = id_map.find(orig_id);
            if (it == id_map.end()) continue;

            int internal_id = it->second;
            if (internal_id < 0 || internal_id >= (int)oracle_probs.size()) continue;

            try {
                oracle_probs[internal_id] = std::stof(row[prob_idx]);
                ++updated_count;
            } catch (...) {
            }
        }
    }
public:
    OracleManager() {}

    /**
     * @brief 加载 ID 映射和 Oracle 概率数据
     * @param dataset_path 数据集根目录
     * @param max_data_vertex_id 数据图中最大的顶点 ID (用于调整 vector 大小)
     */
    void Load(const std::string& dataset_path, int max_data_vertex_id) {
        std::string map_file = dataset_path + "/data_graph/id_mapping.csv";
        std::string post_file = dataset_path + "/csv_data/post.csv";

        // 初始化概率数组，默认值为 0.0
        if (max_data_vertex_id >= 0) {
            oracle_probs.resize(max_data_vertex_id + 1, 0.0f);
        }

        // ==========================================
        // 1. 读取 id_mapping.csv
        // 目标: 建立 orig_id (String) -> internal_id (Int) 的映射
        // ==========================================
        std::unordered_map<std::string, int> orig_to_internal;
        std::ifstream f_map(map_file);
        
        if (!f_map.is_open()) {
            std::cerr << "[Error] Cannot open id_mapping file: " << map_file << std::endl;
            return;
        }

        std::string line;
        // 读取表头并确定列索引 (id_mapping 通常比较标准，但动态查找更安全)
        if (std::getline(f_map, line)) {
            // 简单处理 id_mapping 的表头，通常是 internal_id,orig_id,type
            // 这里为了性能和兼容性，假设 id_mapping 格式相对固定
            // 如果您的 id_mapping 也很复杂，可以使用下面的 parse_csv_line
        }

        while (std::getline(f_map, line)) {
            // id_mapping 通常不含复杂文本，可以用简单的流处理，或者统一用 parse_csv_line
            std::vector<std::string> row = parse_csv_line(line);
            
            // 假设列顺序: internal_id(0), orig_id(1), type(2)
            // 请根据实际 id_mapping.csv 调整索引，或者添加表头解析逻辑
            if (row.size() >= 3) {
                // 简单去除可能的两端空格
                // int internal_id = std::stoi(row[0]);
                // std::string orig_id = row[1];
                // std::string type = row[2];
                
                // 这里假设只关心 Post 类型，根据您的 id_mapping 实际内容调整
                // if (type == "Post") { 
                     try {
                        int i_id = std::stoi(row[0]);
                        orig_to_internal[row[1]] = i_id;
                     } catch (...) {}
                // }
            }
        }
        f_map.close();
        std::cout << "[Info] Loaded ID Mapping. Post count: " << orig_to_internal.size() << std::endl;

        // ==========================================
        // 2. 读取 post.csv
        // 目标: orig_id -> probability -> 存入 vector
        // ==========================================
        std::ifstream f_post(post_file);
        if (!f_post.is_open()) {
            std::cerr << "[Error] Cannot open post file: " << post_file << std::endl;
            return;
        }

        // 2.1 解析表头，动态查找列索引
        int id_col_idx = -1;
        int prob_col_idx = -1;
        
        // 目标列名
        const std::string TARGET_ID_COL = "id:ID";
        // const std::string TARGET_PROB_COL = "ML1_oracle2_probability";
        const std::string TARGET_PROB_COL = "Dist_Beta_U_oracle_prob";

        std::cout<<"[info]" << TARGET_ID_COL <<","<< TARGET_PROB_COL <<std::endl;
        
        if (std::getline(f_post, line)) {
            std::vector<std::string> header = parse_csv_line(line);
            for (size_t i = 0; i < header.size(); ++i) {
                // 去除可能存在的 BOM 头或空白字符
                std::string col = header[i];
                // 简单的清理
                col.erase(0, col.find_first_not_of(" \t\r\n\""));
                col.erase(col.find_last_not_of(" \t\r\n\"") + 1);

                if (col == TARGET_ID_COL) {
                    id_col_idx = i;
                } else if (col == TARGET_PROB_COL) {
                    prob_col_idx = i;
                }
            }
        }

        if (id_col_idx == -1 || prob_col_idx == -1) {
            std::cerr << "[Error] Could not find required columns in post.csv." << std::endl;
            std::cerr << "  Looking for: " << TARGET_ID_COL << " and " << TARGET_PROB_COL << std::endl;
            return;
        }

        // 2.2 读取数据行
        int loaded_count = 0;
        while (std::getline(f_post, line)) {
            std::vector<std::string> row = parse_csv_line(line);

            // 确保行有足够的列
            if ((int)row.size() > std::max(id_col_idx, prob_col_idx)) {
                std::string orig_id = row[id_col_idx];
                std::string prob_str = row[prob_col_idx];

                // 如果这个 orig_id 存在于我们的映射表中
                if (orig_to_internal.count(orig_id)) {
                    int internal_id = orig_to_internal[orig_id];
                    
                    // 边界检查，防止越界
                    if (internal_id >= 0 && internal_id < (int)oracle_probs.size()) {
                        try {
                            float prob = std::stof(prob_str);
                            oracle_probs[internal_id] = prob;
                            loaded_count++;
                        } catch (...) {
                            // 忽略解析错误的数值
                        }
                    }
                }
            }
        }
        f_post.close();
        
        std::cout << "[Info] OracleManager loaded. Probabilities updated for " << loaded_count << " vertices." << std::endl;
    }

    void LoadAggColumn(const std::string& dataset_path,
                   int max_data_vertex_id,
                   const std::string& table,     // "post" or "comment"
                   const std::string& value_col) {
        if (max_data_vertex_id >= 0) {
            agg_values.assign(max_data_vertex_id + 1, 0.0);
        }

        // 1) 读 id_mapping
        std::string map_file = dataset_path + "/data_graph/id_mapping.csv";
        std::unordered_map<std::string, int> orig_to_internal;
        std::ifstream f_map(map_file);
        if (!f_map.is_open()) {
            std::cerr << "[Error] Cannot open id_mapping file: " << map_file << std::endl;
            return;
        }

        std::string line;
        while (std::getline(f_map, line)) {
            std::vector<std::string> row = parse_csv_line(line);
            if (row.size() >= 2) {
                try {
                    int i_id = std::stoi(row[0]);
                    orig_to_internal[row[1]] = i_id;
                } catch (...) {}
            }
        }
        f_map.close();

        // 2) 读对应表
        std::string filepath = dataset_path + "/csv_data/" + table + ".csv";
        int updated = 0;
        std::cout << "[Info] Loading SUM column from " << filepath
                << " col: " << value_col << std::endl;

        LoadDoublesFromCSV(filepath, "id:ID", value_col, orig_to_internal, agg_values, updated);

        std::cout << "[Info] Agg column ready. Updated rows: " << updated << std::endl;
    }

    double GetAggValue(int internal_id) const {
        if (internal_id < 0 || internal_id >= (int)agg_values.size()) return 0.0;
        return agg_values[internal_id];
    }
    
    // void LoadMulti(const std::string& dataset_path, int max_data_vertex_id, 
    //                const std::string& post_oracle_col, 
    //                const std::string& comment_oracle_col) {
        
    //     // 初始化概率向量
    //     if (max_data_vertex_id >= 0) {
    //         oracle_probs.assign(max_data_vertex_id + 1, 0.0f);
    //     }

    //     // 1. 加载 ID Mapping (String -> Internal Int)
    //     std::string map_file = dataset_path + "/data_graph/id_mapping.csv";
    //     std::unordered_map<std::string, int> orig_to_internal;
    //     std::ifstream f_map(map_file);
    //     // ... (此处保留原有的 ID Mapping 加载逻辑，略去重复代码以节省空间) ...
    //     // ...existing code...
    //     if (!f_map.is_open()) { /* Error handling */ return; }
    //     std::string line;
    //     // Skip header logic if needed
    //     while (std::getline(f_map, line)) {
    //          std::vector<std::string> row = parse_csv_line(line);
    //          if (row.size() >= 2) {
    //              try {
    //                 int i_id = std::stoi(row[0]);
    //                 orig_to_internal[row[1]] = i_id;
    //              } catch (...) {}
    //          }
    //     }
    //     f_map.close();
    //     std::cout << "[Info] Loaded ID Mapping size: " << orig_to_internal.size() << std::endl;

    //     int total_updated = 0;

    //     // 2. 加载 Post 数据
    //     if (!post_oracle_col.empty()) {
    //         std::string post_file = dataset_path + "/csv_data/post.csv";
    //         std::cout << "[Info] Loading Post Oracle from " << post_file << " col: " << post_oracle_col << std::endl;
    //         LoadProbabilitiesFromCSV(post_file, "id:ID", post_oracle_col, orig_to_internal, total_updated);
    //     }

    //     // 3. 加载 Comment 数据
    //     if (!comment_oracle_col.empty()) {
    //         std::string comment_file = dataset_path + "/csv_data/comment.csv"; // 假设文件名是 comment.csv
    //         std::cout << "[Info] Loading Comment Oracle from " << comment_file << " col: " << comment_oracle_col << std::endl;
    //         LoadProbabilitiesFromCSV(comment_file, "id:ID", comment_oracle_col, orig_to_internal, total_updated);
    //     }

    //     std::cout << "[Info] OracleManager Ready. Total vertices with probability: " << total_updated << std::endl;
    // }
    

    void LoadMulti(const std::string& dataset_path, int max_data_vertex_id, 
                   const std::string& table1_name, const std::string& table1_oracle_col, 
                   const std::string& table2_name, const std::string& table2_oracle_col) {
        
        // 初始化概率向量
        if (max_data_vertex_id >= 0) {
            oracle_probs.assign(max_data_vertex_id + 1, 0.0f);
        }

        // 1. 加载 ID Mapping (String -> Internal Int)
        std::string map_file = dataset_path + "/data_graph/id_mapping.csv";
        std::unordered_map<std::string, int> orig_to_internal;
        std::ifstream f_map(map_file);
        // ... (此处保留原有的 ID Mapping 加载逻辑，略去重复代码以节省空间) ...
        // ...existing code...
        if (!f_map.is_open()) { /* Error handling */ return; }
        std::string line;
        // Skip header logic if needed
        while (std::getline(f_map, line)) {
             std::vector<std::string> row = parse_csv_line(line);
             if (row.size() >= 2) {
                 try {
                    int i_id = std::stoi(row[0]);
                    orig_to_internal[row[1]] = i_id;
                 } catch (...) {}
             }
        }
        f_map.close();
        std::cout << "[Info] Loaded ID Mapping size: " << orig_to_internal.size() << std::endl;

        int total_updated = 0;

        // // 2. 加载 Post 数据
        // if (!post_oracle_col.empty()) {
        //     std::string post_file = dataset_path + "/csv_data/post.csv";
        //     std::cout << "[Info] Loading Post Oracle from " << post_file << " col: " << post_oracle_col << std::endl;
        //     LoadProbabilitiesFromCSV(post_file, "id:ID", post_oracle_col, orig_to_internal, total_updated);
        // }

        // // 3. 加载 Comment 数据
        // if (!comment_oracle_col.empty()) {
        //     std::string comment_file = dataset_path + "/csv_data/comment.csv"; // 假设文件名是 comment.csv
        //     std::cout << "[Info] Loading Comment Oracle from " << comment_file << " col: " << comment_oracle_col << std::endl;
        //     LoadProbabilitiesFromCSV(comment_file, "id:ID", comment_oracle_col, orig_to_internal, total_updated);
        // }

        // 2. 加载表 1 数据 (例如 post 或 review)
        if (!table1_oracle_col.empty()) {
            std::string file1 = dataset_path + "/csv_data/" + table1_name + ".csv";
            std::cout << "[Info] Loading Oracle 1 from " << file1 << " col: " << table1_oracle_col << std::endl;
            LoadProbabilitiesFromCSV(file1, "id:ID", table1_oracle_col, orig_to_internal, total_updated);
        }

        // 3. 加载表 2 数据 (例如 comment 或 product)
        if (!table2_oracle_col.empty()) {
            std::string file2 = dataset_path + "/csv_data/" + table2_name + ".csv";
            std::cout << "[Info] Loading Oracle 2 from " << file2 << " col: " << table2_oracle_col << std::endl;
            LoadProbabilitiesFromCSV(file2, "id:ID", table2_oracle_col, orig_to_internal, total_updated);
        }

        std::cout << "[Info] OracleManager Ready. Total vertices with probability: " << total_updated << std::endl;
    }
 

    /**
     * @brief 检查 Oracle 条件
     * @param data_node_id 数据图中的内部顶点 ID
     * @param threshold 阈值 (默认 0.5)
     * @return true 如果概率 > 阈值
     */
    bool CheckOracle(int data_node_id, float threshold = 0.5) {
        if (data_node_id < 0 || data_node_id >= (int)oracle_probs.size()) return false;
        return oracle_probs[data_node_id] > threshold;
    }
};

} // namespace CardinalityEstimation
} // namespace GraphLib