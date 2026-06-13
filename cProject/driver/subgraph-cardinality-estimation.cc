#include <iostream>
#include <set>
#include "Base/Metrics.h"
#include "Base/Timer.h"
#include "DataStructure/Graph.h"
#include "SpecialSubgraphs/SmallCycle.h"
#include "SubgraphMatching/DataGraph.h"
#include "SubgraphMatching/PatternGraph.h"
#include "SubgraphMatching/CandidateSpace.h"
#include "SubgraphMatching/CandidateFilter.h"
#include "SubgraphCounting/Option.h"
#include "SubgraphCounting/CardinalityEstimation.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <iomanip>
#include <filesystem>
#include <regex>
#include <unordered_set>
#include <future>
#include <thread>
#include <algorithm>
#include <mutex>
#include <atomic>

using namespace std;
using namespace GraphLib;
namespace fs = std::filesystem;
struct FastestOJob {
    size_t job_index;
    int run_id;
    double frac;
    int budget_n;
    int oracle_cost;
};

struct FastestOResult {
    size_t job_index;
    int run_id;
    double frac;
    int budget_n;
    double t_true;
    double est;
    int unique_nodes;
    int oracle_cost;
};

std::set<std::string> scientific_type_results = {"#CandTree"};
std::set<std::string> double_type_results = {
    "Truth", "Est", "logQError", "CSBuildTime", "TreeCountTime", "TreeSampleTime", "GraphSampleTime", "QueryTime","#UniqueOracleNodes"
};
std::set<std::string> longlong_type_results = {};
std::vector<std::string> print_order = {
    "#CSVertex", "#CSEdge", "#CandTree", "#TreeTrials", "#TreeSuccess","#UniqueOracleNodes","Truth", "Est", "logQError",
    "CSBuildTime", "TreeCountTime", "TreeSampleTime", "GraphSampleTime", "QueryTime"
};
std::vector<dict> results;
std::string query_path;
Timer timer;
std::vector<PatternGraph*> pattern_graphs;
std::deque<std::string> query_names;
std::unordered_map<std::string, double> true_cnt;
double total_time = 0.0;
// 用于存储从文件读取的，每个查询对应的核心标签列表
std::unordered_map<std::string, std::vector<int>> query_to_core_labels;
std::unordered_map<std::string, std::vector<int>> query_to_core_query_nodes;
std::unordered_map<std::string, std::vector<int>> query_to_predicate_labels;

// 定义一个全局或静态的 map 来缓存预算
std::unordered_map<std::string, int> g_budget_cache;
bool g_budget_loaded = false;
bool g_budget_curve_loaded = false;
std::unordered_map<std::string, std::unordered_map<std::string, int>> g_budget_curve_cache;
std::string agg_func_str = "count"; // "count" or "sum"
std::string sum_table;             // "post" or "comment"
std::string sum_col;               // 列名，如 "score"
int sum_label = -1;                // 语义B：指定 3/4/5 中一个，或 post 的 2
std::mutex g_oracle_mutex;

// 辅助函数：去除文件名后缀
// std::string get_raw_name(const std::string& filename) {
//     std::string basename = std::filesystem::path(filename).filename().string();
//     size_t lastindex = basename.find_last_of(".");
//     return (lastindex == std::string::npos) ? basename : basename.substr(0, lastindex);
// }
std::string get_filename_only(const std::string& filepath) {
    return std::filesystem::path(filepath).filename().string();
}
std::string get_filename_without_extension(const std::string& filename) {
    std::string basename = std::filesystem::path(filename).filename().string();
    size_t lastindex = basename.find_last_of(".");
    // 如果找到了点，并且不是在开头（避免隐藏文件问题），则截断
    if (lastindex != std::string::npos && lastindex > 0) { 
        return basename.substr(0, lastindex);
    }
    return basename;
}

// [MOD-2] 归一化 budget_frac（去掉空格/引号/尾随0）
std::string normalize_budget_frac_string(const std::string& raw) {
    std::string s = raw;
    s.erase(0, s.find_first_not_of(" \t\r\n\""));
    s.erase(s.find_last_not_of(" \t\r\n\"") + 1);
    if (s.empty()) return s;

    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s.empty() ? "0" : s;
}

std::string normalize_budget_frac(double frac) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << frac;
    return normalize_budget_frac_string(oss.str());
}

// [MOD-2] 读取预算曲线文件（FOIS_rs_POSS_budget_curve.csv）
void load_budget_curve_cache(const std::string& curve_path, const std::string& target_method = "FOIS_nrs") {
    if (g_budget_curve_loaded) return;

    if (!std::filesystem::exists(curve_path)) {
        std::cerr << "[Warning] Budget curve file not found: " << curve_path << std::endl;
        return;
    }

    std::ifstream file(curve_path);
    if (!file.is_open()) {
        std::cerr << "[Warning] Cannot open budget curve file: " << curve_path << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // header

    // CSV 结构:
    // [0]query_basename, [1]run_id, [2]budget_frac, [3]budget_n, [4]T_true,
    // [5]T_hat, [6]Qerror, [7]n_post, [8]n_comment, [9]oracle_cost, [10]method
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> row;
        while (std::getline(ss, segment, ',')) {
            row.push_back(segment);
        }

        if (row.size() >= 11) {
            std::string method = row[10];
            if (method != target_method) continue;

            std::string q_name = row[0];
            std::string clean_key = get_filename_without_extension(q_name);
            std::string frac_key = normalize_budget_frac_string(row[2]);

            try {
                int oracle_cost = std::stoi(row[9]);
                double gamma = 1.2; 
                int calibrated_cost = static_cast<int>(std::ceil(gamma * oracle_cost));
                g_budget_curve_cache[clean_key][frac_key] = calibrated_cost;
            } catch (...) {
                // ignore bad rows
            }
        }
    }
    g_budget_curve_loaded = true;
    std::cout << "[Info] Loaded budget curve entries from " << curve_path << std::endl;
}

// [MOD-2] 按 (query, budget_frac) 查询 oracle_cost
int get_oracle_cost_from_curve(const std::string& query_path, double budget_frac) {
    std::string key = get_filename_without_extension(query_path);
    std::string frac_key = normalize_budget_frac(budget_frac);

    auto it = g_budget_curve_cache.find(key);
    if (it == g_budget_curve_cache.end()) return -1;

    auto jt = it->second.find(frac_key);
    if (jt == it->second.end()) return -1;

    return jt->second;
}
// 改进后的加载函数：只读取一次文件
void load_budget_cache(const std::string& summary_path, const std::string& target_method = "FOIS_nrs") {
    if (g_budget_loaded) return; 

    if (!std::filesystem::exists(summary_path)) {
        std::cerr << "[Warning] Summary file not found: " << summary_path << std::endl;
        return;
    }

    std::ifstream file(summary_path);
    if (!file.is_open()) {
        std::cerr << "[Warning] Cannot open summary file: " << summary_path << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header

    // CSV 结构: 
    // [0]query_index, [1]query_basename, [2]gt_match_col, [3]T_true, [4]method, [5]T_hat, [6]Qerror, [7]n_post, [8]n_comment
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> row;
        while(std::getline(ss, segment, ',')) {
            row.push_back(segment);
        }

        if (row.size() >= 9) { // 确保有 n_post 和 n_comment 列
            std::string q_name = row[1];
            std::string clean_key = get_filename_without_extension(q_name);
            std::string method = row[4];
            
            if (method == target_method) {
                try {
                    int n_post_val = std::stoi(row[7]);
                    int n_comment_val = std::stoi(row[8]);
                    // [Key Change] 这里的 budget 是两者之和
                    g_budget_cache[clean_key] = n_post_val + n_comment_val; 
                } catch (...) {
                    // 忽略解析错误的行
                }
            }
        }
    }
    g_budget_loaded = true;
    std::cout << "[Info] Loaded " << g_budget_cache.size() << " budget entries (sum of post+comment) from summary file." << std::endl;
}

// 改进后的获取函数：纯内存查表
int get_budget_limit_optimized(const std::string& query_path) {
    // 1. 获取 raw_name (e.g., "query_dense_1_1")
    std::string raw_name = get_filename_without_extension(query_path);

    // 2. 查表
    auto it = g_budget_cache.find(raw_name);
    if (it != g_budget_cache.end()) {
        return it->second;
    }
    return -1; // Not found
}

void save_sampled_node_count(const std::string& csv_path, 
                             const std::string& query_name, 
                             int post_cnt) {
    bool file_exists = std::filesystem::exists(csv_path);
    std::ofstream csv_out(csv_path, std::ios::app);
    
    if (!csv_out.is_open()) {
        std::cerr << "[Error] Cannot open " << csv_path << " for appending." << std::endl;
        return;
    }

    // 如果文件不存在，先写入表头
    if (!file_exists) {
        csv_out << "query_name,method,post_sampled_cnt,comment_sampled_cnt\n";
    }

    // query_name 需要去掉可能的路径，只保留文件名
    std::string basename = std::filesystem::path(query_name).filename().string();
    // 有些系统可能带 .graph 后缀，通常 csv 里存的是带后缀或不带的，根据你提供的样例是 query_cycle_4_0 (无后缀)
    // 这里为了匹配你给的样例 "query_cycle_4_0"，去掉 .graph
    size_t lastindex = basename.find_last_of("."); 
    std::string raw_name = (lastindex == std::string::npos) ? basename : basename.substr(0, lastindex);

    // 写入数据
    csv_out << raw_name << ","
            << "baseline2_graph_only" << ","
            << post_cnt << ","
            << "0" << "\n"; // comment_sampled_cnt 固定为 0

    csv_out.close();
    std::cout << "[Info] Appended node counts to " << csv_path << std::endl;
}



void append_to_results_summary(const std::string& csv_path, 
                               const std::string& query_name, 
                               double t_hat_mean,
                               int unique_nodes_count) {
    // std::string basename = std::filesystem::path(query_name).filename().string();
    std::string basename = get_filename_without_extension(query_name);
    // 1. 读取现有文件以查找元数据 (query_index, gt_match_col, T_true)
    std::ifstream csv_in(csv_path);
    std::string line;
    std::string found_query_index = "-1";
    std::string found_gt_match_col = "unknown";
    double found_t_true = 0.0;
    bool found = false;

    if (csv_in.is_open()) {
        std::getline(csv_in, line); // header
        while (std::getline(csv_in, line)) {
            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> row;
            while(std::getline(ss, segment, ',')) { // bug fix: split on comma
                row.push_back(segment);
            }
            if (row.size() >= 4) {
                if (row[1] == basename) {
                    found_query_index = row[0];
                    found_gt_match_col = row[2];
                    try {
                        found_t_true = std::stod(row[3]);
                    } catch (...) { found_t_true = 0.0; }
                    found = true;
                    break; 
                }
            }
        }
        csv_in.close();
    }

    if (!found) {
        std::cerr << "[Warning] Could not find existing metadata for " << basename 
                  << " in summary file. Appending with default values." << std::endl;
    }

    // 2. 计算 Qerror
    double q_error = 0.0;
    if (found_t_true != 0.0) {
        q_error = std::abs(t_hat_mean - found_t_true) / found_t_true;
    } else if (found && found_t_true == 0.0 && t_hat_mean > 0) {
         q_error = 1.0; 
    }

    // 3. 追加写入
    bool file_exists = std::filesystem::exists(csv_path);
    std::ofstream csv_out(csv_path, std::ios::app);
    if (!csv_out.is_open()) {
        std::cerr << "[Error] Cannot open " << csv_path << " for appending." << std::endl;
        return;
    }

    // [New] 写入符合要求的表头
    if (!file_exists) {
        csv_out << "query_index,query_basename,gt_match_col,T_true,method,T_hat,Qerror,n_post,n_comment\n";
    }

    // Format: query_index, query_basename, gt_match_col, T_true, method, T_hat, Qerror, n_post, n_comment
    csv_out << found_query_index << ","
            << basename << ","
            << found_gt_match_col << ","
            << std::fixed << std::setprecision(1) << found_t_true << ","
            << "FaSTestO" << ","
            << std::setprecision(4) << t_hat_mean << ","
            << std::setprecision(8) << q_error << ","
            << unique_nodes_count << "," // 这里写入总消耗 (n_post位置)
            << "0" << "\n";              // n_comment 暂留空或写0

    csv_out.close();
    std::cout << "[Info] Appended summary result to " << csv_path << std::endl;
}


// 生成 FastestO 预算曲线（输出格式与 FOIS_rs_POSS_budget_curve.csv 完全一致）
void append_fastesto_budget_curve_row(const std::string& csv_path,
                                      const std::string& query_name,
                                      int run_id,
                                      double budget_frac,
                                      int budget_n,
                                      double t_true,
                                      double t_hat,
                                      int n_post,
                                      int n_comment,
                                      int oracle_cost,
                                      const std::string& method) {
    bool file_exists = std::filesystem::exists(csv_path);
    bool need_header = !fs::exists(csv_path) || (fs::exists(csv_path) && fs::file_size(csv_path) == 0);
    std::ofstream csv_out(csv_path, std::ios::app);
    if (!csv_out.is_open()) {
        std::cerr << "[Error] Cannot open " << csv_path << " for appending." << std::endl;
        return;
    }

    // 写入表头（与 FOIS_rs_POSS_budget_curve.csv 一致）
    if (need_header) {
        csv_out << "query_basename,run_id,budget_frac,budget_n,T_true,T_hat,Qerror,n_post,n_comment,oracle_cost,method\n";
    }

    // 计算 Qerror
    double q_error = 0.0;
    if (t_true != 0.0) {
        q_error = std::abs(t_hat - t_true) / t_true;
    } else if (t_hat > 0.0) {
        q_error = 1.0;
    }

    // query_basename 保留 .graph 后缀（与样例一致）
    std::string query_basename = get_filename_only(query_name);

    // 输出格式控制：budget_frac 保留 0.05 / 0.1 的风格，T_true 保留一位小数
    csv_out << query_basename << ","
            << run_id << ","
            << std::defaultfloat << std::setprecision(2) << budget_frac << ","
            << budget_n << ","
            << std::fixed << std::setprecision(1) << t_true << ","
            << std::defaultfloat << std::setprecision(15) << t_hat << ","
            << std::setprecision(16) << q_error << ","
            << n_post << ","
            << n_comment << ","
            << oracle_cost << ","
            << method << "\n";

    csv_out.close();
}

void read_core_labels(const std::string& dataset) {
    std::string config_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/data_graph/user_custom_labels.txt";
    std::cout << "[Info] Reading core labels configuration from: " << config_path << std::endl;

    std::ifstream config_in(config_path);
    if (!config_in.is_open()) {
        std::cout << "[Info] Core labels configuration file not found. Running in global estimation mode only." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(config_in, line)) {
        if (line.empty() || line[0] == '#') { // 忽略空行和注释行
            continue;
        }

        std::stringstream ss(line);
        std::string query_filename;
        ss >> query_filename;

        if (query_filename.empty()) {
            continue;
        }

        // 构造与 query_names 中一致的完整路径作为 map 的 key
        std::string full_query_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/query_graph/" + query_filename;

        std::vector<int> labels;
        int label;
        while (ss >> label) {
            labels.push_back(label);
        }

        if (!labels.empty()) {
            query_to_core_labels[full_query_path] = labels;
        }
    }
    std::cout << "[Info] Loaded core label configurations for " << query_to_core_labels.size() << " queries." << std::endl;
}

void read_core_nodes_config(const std::string& dataset) {
    std::string config_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/data_graph/core_nodes_config.json";
    std::cout << "[Info] Reading core nodes configuration from: " << config_path << std::endl;

    std::ifstream fin(config_path);
    if (!fin.is_open()) {
        std::cout << "[Warning] core_nodes_config.json not found. Fallback to old behavior." << std::endl;
        return;
    }

    std::string text((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    fin.close();

    query_to_core_query_nodes.clear();
    query_to_predicate_labels.clear();

    std::regex query_block_re("\"([^\"]+\\.graph)\"\\s*:\\s*\\{([\\s\\S]*?)\\}");
    std::regex label_array_re("\"(\\d+)\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::regex int_re("-?\\d+");

    auto qb_begin = std::sregex_iterator(text.begin(), text.end(), query_block_re);
    auto qb_end = std::sregex_iterator();

    for (auto it = qb_begin; it != qb_end; ++it) {
        std::string query_filename = (*it)[1].str();
        std::string block = (*it)[2].str();

        std::string full_query_path =
            "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/query_graph/" + query_filename;

        std::vector<int> core_query_nodes;
        std::vector<int> predicate_labels;
        std::unordered_set<int> seen_nodes;
        std::unordered_set<int> seen_labels;

        auto la_begin = std::sregex_iterator(block.begin(), block.end(), label_array_re);
        auto la_end = std::sregex_iterator();

        for (auto jt = la_begin; jt != la_end; ++jt) {
            int label = std::stoi((*jt)[1].str());
            std::string arr = (*jt)[2].str();

            if (!seen_labels.count(label)) {
                predicate_labels.push_back(label);
                seen_labels.insert(label);
            }

            auto n_begin = std::sregex_iterator(arr.begin(), arr.end(), int_re);
            auto n_end = std::sregex_iterator();
            for (auto kt = n_begin; kt != n_end; ++kt) {
                int qid = std::stoi((*kt).str());
                if (!seen_nodes.count(qid)) {
                    core_query_nodes.push_back(qid);
                    seen_nodes.insert(qid);
                }
            }
        }

        std::sort(core_query_nodes.begin(), core_query_nodes.end());

        if (!core_query_nodes.empty()) {
            query_to_core_query_nodes[full_query_path] = core_query_nodes;
        }
        if (!predicate_labels.empty()) {
            query_to_predicate_labels[full_query_path] = predicate_labels;
        }
    }

    std::cout << "[Info] Loaded core query-node configs for " << query_to_core_query_nodes.size() << " queries." << std::endl;
    std::cout << "[Info] Loaded predicate-label configs for " << query_to_predicate_labels.size() << " queries." << std::endl;
}

void read_ans(const std::string& dataset) {
    std::string ans_file_name = query_path;
    cout << ans_file_name << endl;
    std::ifstream ans_in(ans_file_name);
    while (!ans_in.eof()) {
        std::string name, t, c;
        ans_in >> name >> t >> c;
        if (name.empty() || c.empty()) continue;
        // name = "../dataset/"+dataset+"/query_graph/"+name;
        name = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/query_graph/"+name;
        true_cnt[name] = stod(c);
        query_names.push_back(name);
    }
}
void write_instance_results_to_csv(
    const std::string& csv_path,
    const std::string& query_name,
    const std::pair<double, std::map<std::vector<int>, double>>& result_pair) 
{
    std::ofstream csv_out(csv_path, std::ios::app);
    if (!csv_out.is_open()) { /* ... error handling ... */ return; }

    double global_estimate = result_pair.first;
    const auto& instance_freqs = result_pair.second;

    std::filesystem::path p(query_name);
    std::string base_filename = p.filename().string();

    long long instance_counter = 0; // 用于生成唯一的 instance_id

    if (instance_freqs.empty()) {
        // (可以选择不写入任何内容，或者写入一个标记行)
    } else {
        for (auto const& [nodes, freq] : instance_freqs) {
            instance_counter++; // 每个新实例，ID加一
            for (size_t i = 0; i < nodes.size(); ++i) {
                csv_out << base_filename << ","          // query_name
                        << instance_counter << ","       // instance_id
                        << (i + 1) << ","                // core_node_index (从1开始)
                        << nodes[i] << ","               // node_id
                        << std::fixed << std::setprecision(4) << freq << "," // estimateW
                        << std::fixed << std::setprecision(4) << global_estimate << "\n"; // global_estimateW
            }
        }
    }
    csv_out.close();
    std::cout << "[Info] Appended instance results for " << base_filename << " to CSV in long format." << std::endl;
}


void read_filter_option(const std::string& opt, const std::string &filter, CardinalityEstimation::CardEstOption& option) {
    if (opt.substr(2) == "STRUCTURE") {
        if (filter == "X")
            option.structure_filter = SubgraphMatching::NO_STRUCTURE_FILTER;
        else if (filter == "3")
            option.structure_filter = SubgraphMatching::TRIANGLE_SAFETY;
        else if (filter == "4")
            option.structure_filter = SubgraphMatching::FOURCYCLE_SAFETY;
    }
}

int32_t main(int argc, char *argv[]) {
    std::string dataset = "dataset_three";
    std::string parent_dataset = "amazon_data";
    CardinalityEstimation::CardEstOption opt;
    bool run_with_predicate = false;
    // ===  FastestO 预算曲线模式 ===
    bool run_fastesto_budget_curve = false;
    int fastesto_runs = 1; // 默认只跑一次（可通过 CLI 修改）
    // std::vector<double> fastesto_budget_fracs = {0.01,0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> fastesto_budget_fracs = {0.01,0.05, 0.075,0.1, 0.15, 0.2,0.3, 0.4, 0.5,0.6,0.7,0.8,0.9};
    std::string fastesto_budget_curve_path;

    std::string post_oracle_col = "ML1_oracle2_probability"; 
    std::string comment_oracle_col = "ML1_oracle2_probability";
    std::string multi_proxy_prob = "ML1_proxy4b_probability";
    std::string budget_curve_path;
    std::string oracle_table1 = "post";     // 默认兼容 parler
    std::string oracle_table2 = "comment";  

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                case 'd':
                    dataset = argv[i + 1];
                    break;
                case 'q':
                    query_path = argv[i + 1];
                    break;
                case 'K':
                    opt.ub_initial = atoi(argv[i + 1]);
                    break;
                case '-':
                    // 在解析命令行参数的循环里加入：
                    if (std::string(argv[i]) == "--ROOT_LABEL") {
                        opt.root_label = atoi(argv[++i]);
                        std::cout << "[CLI] Root label set to " << opt.root_label << std::endl;
                    } else if (std::string(argv[i]) == "--ROOT_INDEX" ||
                            std::string(argv[i]) == "--ROOT_QUERY_INDEX" ||
                            std::string(argv[i]) == "--ROOT_IDX") {
                        opt.root_query_index = atoi(argv[++i]);
                        std::cout << "[CLI] Root query index set to " << opt.root_query_index << std::endl;
                    } else if (std::string(argv[i]) == "--SAMPLE_BUDGET") {
                        opt.sample_budget = atoi(argv[++i]);
                        std::cout << "[CLI] Sample budget set to " << opt.sample_budget << std::endl;
                    }else if (std::string(argv[i]) == "--ESTIMATE_WITH_PREDICATE") {
                        run_with_predicate = true;
                        std::cout << "[CLI] Mode: Estimate with Predicate enabled." << std::endl;
                    }else if (std::string(argv[i]) == "--POST_ORACLE_COL") {
                        post_oracle_col = argv[++i];
                        std::cout << "[CLI] Post Oracle Col: " << post_oracle_col << std::endl;
                    } 
                    else if (std::string(argv[i]) == "--COMMENT_ORACLE_COL") {
                        comment_oracle_col = argv[++i];
                        std::cout << "[CLI] Comment Oracle Col: " << comment_oracle_col << std::endl;
                    }else if (std::string(argv[i]) == "--MULTI_PROXY_PROB") {
                        multi_proxy_prob = argv[++i];
                        std::cout << "[CLI] Multi Proxy Prob Dir Name: " << multi_proxy_prob << std::endl;
                    }else if (std::string(argv[i]) == "--FASTESTO_BUDGET_CURVE") {
                        run_fastesto_budget_curve = true;
                        std::cout << "[CLI] FastestO budget curve enabled." << std::endl;
                    } else if (std::string(argv[i]) == "--FASTESTO_BUDGET_CURVE_OUT") {
                        fastesto_budget_curve_path = argv[++i];
                        std::cout << "[CLI] FastestO budget curve output: " << fastesto_budget_curve_path << std::endl;
                    } else if (std::string(argv[i]) == "--FASTESTO_RUNS") {
                        fastesto_runs = std::max(1, atoi(argv[++i]));
                        std::cout << "[CLI] FastestO runs set to " << fastesto_runs << std::endl;
                    }else if (std::string(argv[i]) == "--BUDGET_CURVE_IN") {
                        budget_curve_path = argv[++i];
                        std::cout << "[CLI] Budget curve input: " << budget_curve_path << std::endl;
                    } else if (std::string(argv[i]) == "--AGG_FUNC") {
                        agg_func_str = argv[++i];
                    } else if (std::string(argv[i]) == "--SUM_TABLE") {
                        sum_table = argv[++i];
                    } else if (std::string(argv[i]) == "--SUM_COL") {
                        sum_col = argv[++i];
                    } else if (std::string(argv[i]) == "--SUM_LABEL") {
                        sum_label = atoi(argv[++i]);
                    } else if (std::string(argv[i]) == "--ORACLE_TABLE1") {
                        oracle_table1 = argv[++i];
                    } else if (std::string(argv[i]) == "--ORACLE_TABLE2") {
                        oracle_table2 = argv[++i];
                    }else {
                        read_filter_option(std::string(argv[i]), std::string(argv[i+1]), opt);
                    }
                    break;
                default:
                    break;
            }
        }
    }
    std::cout << "[info] SAMPLE_BUDGET:" <<opt.sample_budget<< std::endl;
    std::string sv_out_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/results/in_estimateW_result.txt";
    std::string results_dir = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/results/";
    std::string sampled_node_count_path = results_dir + "efficiency/sampled_node_count.csv";
    std::string results_summary_path = results_dir + "results_summary_FaSTestO.csv";
    std::string summary_run1_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/results/result_summarys/" + multi_proxy_prob + "/results_summary_run_1.csv";
    std::vector<std::pair<std::string, double>> basic_estimates_for_json;
    CardinalityEstimation::AggFunc agg_func = CardinalityEstimation::AGG_COUNT;
    if (agg_func_str == "sum") agg_func = CardinalityEstimation::AGG_SUM;
    
    // --- Step 0. 清空旧文件内容 ---
    std::ofstream clear_file(sv_out_path, std::ios::out | std::ios::trunc);
    if (clear_file.is_open()) {
        clear_file.close();
        std::cout << "[Info] Cleared old estimate file: " << sv_out_path << std::endl;
    } else {
        std::cerr << "[Error] Cannot open file to clear: " << sv_out_path << std::endl;
    }
    if (budget_curve_path.empty()) {
        budget_curve_path = results_dir + "efficiency/FOIS_rs_POSS_budget_curve_fast.csv";
        std::cout << "[Info] budget_curve_path: " << budget_curve_path << std::endl;
    }
    if (query_path.empty()) {        // query_path = "../dataset/"+dataset+"/"+dataset+"_ans.txt";
        query_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/ground_truth/parler_ans.txt";
    }
    
    if (fastesto_budget_curve_path.empty()) {
        fastesto_budget_curve_path = results_dir + "efficiency/FastestO_budget_curve.csv";
    }

    // 若开启 FastestO budget curve，则清空旧文件
    if (run_fastesto_budget_curve) {
        std::ofstream clear_curve_file(fastesto_budget_curve_path, std::ios::out | std::ios::trunc);
        if (clear_curve_file.is_open()) {
            clear_curve_file.close();
            std::cout << "[Info] Cleared old FastestO curve file: " << fastesto_budget_curve_path << std::endl;
        } else {
            std::cerr << "[Error] Cannot open file to clear: " << fastesto_budget_curve_path << std::endl;
        }
    }

    std::cout << "[info] args:" <<argv<< std::endl;
    std::cout << "[info] summary_run1_path:" <<summary_run1_path<< std::endl;
    std::string data_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/data_graph/parler.graph";
    read_ans(dataset);
    read_core_nodes_config(dataset);

    // --- 准备CSV文件的逻辑保持在main函数中，因为它只需执行一次 ---
    std::string ins_csv_out_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset + "/results/ins_estimateW_result.csv";


    size_t max_core_labels = 0;
    if (!query_to_core_query_nodes.empty()) {
        for (const auto& pair : query_to_core_query_nodes) {
            if (pair.second.size() > max_core_labels) {
                max_core_labels = pair.second.size();
            }
        }
    }
    if (max_core_labels == 0) max_core_labels = 1;
    

    std::ofstream clear_csv_file(ins_csv_out_path, std::ios::out | std::ios::trunc);
    if (clear_csv_file.is_open()) {
        clear_csv_file << "query_name,instance_id,core_node_index,node_id,estimateW,global_estimateW\n";
        clear_csv_file.close();
    }


    DataGraph D;
    D.LoadLabeledGraph(data_path);
    D.Preprocess();
    opt.MAX_QUERY_VERTEX = 12;
    opt.MAX_QUERY_EDGE = 4;
    pattern_graphs.resize(query_names.size());
    for (int i = 0; i < query_names.size(); i++) {
        std::string query_name = query_names[i];
        pattern_graphs[i] = new PatternGraph();
        pattern_graphs[i]->LoadLabeledGraph(query_name);
        pattern_graphs[i]->ProcessPattern(D);
        pattern_graphs[i]->EnumerateLocalTriangles();
        pattern_graphs[i]->EnumerateLocalFourCycles();
        opt.MAX_QUERY_VERTEX = std::max(opt.MAX_QUERY_VERTEX, pattern_graphs[i]->GetNumVertices());
        opt.MAX_QUERY_EDGE = std::max(opt.MAX_QUERY_EDGE, pattern_graphs[i]->GetNumEdges());
    }
    std::cout <<"[infor] root label"<< opt.root_label<< std::endl;
    if (opt.structure_filter >= SubgraphMatching::FOURCYCLE_SAFETY) {
        D.EnumerateLocalFourCycles();
    }
    if (opt.structure_filter >= SubgraphMatching::TRIANGLE_SAFETY) {
        D.EnumerateLocalTriangles();
    }
    CardinalityEstimation::FaSTestCardinalityEstimation estimator(&D, opt);
    std::cout << "[info] Query Size:" << pattern_graphs.size()<< std::endl;
    std::string dataset_base_path = "/home/wangshuo/resource/datasets/amazon_data/" + dataset;

    if (agg_func == CardinalityEstimation::AGG_SUM) {
        if (sum_table.empty() || sum_col.empty() || sum_label < 0) {
            std::cerr << "[Error] SUM requires --SUM_TABLE post|comment, --SUM_COL <col>, --SUM_LABEL <label>\n";
            return 1;
        }
        estimator.InitAggColumn(dataset_base_path, sum_table, sum_col);
    }

    if (run_with_predicate) {
        // 假设数据集路径结构是标准的
         
        std::cout << "[Info] Initializing Oracle Manager from: " << dataset_base_path << std::endl;
        
        // estimator.InitOracle(dataset_base_path);
        // estimator.InitMultiOracle(dataset_base_path, post_oracle_col, comment_oracle_col);
        estimator.InitMultiOracle(dataset_base_path, oracle_table1, post_oracle_col, oracle_table2, comment_oracle_col);
        load_budget_cache(summary_run1_path, "FOIS_nrs");
        load_budget_curve_cache(budget_curve_path, "8_POSSA");
    }
    for (int i = 0; i < pattern_graphs.size(); i++) {
        PatternGraph* P = pattern_graphs[i];
        std::string query_name = query_names[i];
        bool has_custom_labels = query_to_core_labels.count(query_name);
        bool has_core_nodes_cfg = query_to_core_query_nodes.count(query_name);
        bool has_predicate_labels_cfg = query_to_predicate_labels.count(query_name);

        // 检查当前查询是否在配置文件中指定了核心标签
        if (!run_with_predicate&&has_core_nodes_cfg) {  
            // --- 如果指定了，运行核心实例频率估计 ---
            // // 1. 调用估计函数
            const auto& core_query_nodes = query_to_core_query_nodes.at(query_name);
            std::cout << "\nStart Processing " << query_name << " for core instances with query nodes: ";
            for (int qid : core_query_nodes) std::cout << qid << " ";
            std::cout << std::endl;

            auto result_pair = (agg_func == CardinalityEstimation::AGG_COUNT)
                ? estimator.EstimateCoreInstances(P, core_query_nodes)
                : estimator.EstimateCoreInstancesAgg(P, core_query_nodes, agg_func, sum_label);

            // 解包pair得到全局估计值和实例map
            double global_estimate = result_pair.first;
            const auto& instance_freqs = result_pair.second;
            
            // 2. 调用新的写入函数
            write_instance_results_to_csv(ins_csv_out_path, query_name, result_pair);
            // 打印结果
            std::cout << "Global Estimated Value: " << std::fixed << std::setprecision(2) << global_estimate << std::endl;
            std::cout << query_name << " Finished!\n" << std::endl;
            fflush(stdout);
            basic_estimates_for_json.push_back({query_name, global_estimate});
        }else {
            double est = 0.0;
            if (run_with_predicate) {
                std::cout << "\nStart Processing (Multi-Predicate) " << query_name << std::endl;
                
                // std::vector<int> target_labels;
                std::vector<int> target_query_nodes;
                if (has_predicate_labels_cfg) {
                    // target_labels = query_to_predicate_labels.at(query_name);
                    // std::cout << "[Info] Using predicate labels: ";
                    // for (int t : target_labels) std::cout << t << " ";
                    // std::cout << std::endl;
                    target_query_nodes = query_to_core_query_nodes.at(query_name);
                    std::cout << "[Info] Using explicit query nodes for predicates: ";
                    for (int t : target_query_nodes) std::cout << t << " ";
                    std::cout << std::endl;
                }else if (has_predicate_labels_cfg) {
                    // 保留 fallback，但实际执行如果没映射仍有风险
                    std::vector<int> t_labels = query_to_predicate_labels.at(query_name);
                    for (int i = 0; i < P->GetNumVertices(); ++i) {
                        int converted_l = P->GetVertexLabel(i); // 注意：由于有可能获取转译label导致不匹配，这步如果是0可以看运气
                        if (std::find(t_labels.begin(), t_labels.end(), converted_l) != t_labels.end()) {
                            target_query_nodes.push_back(i);
                        }
                    }
                } else {
                    // if (opt.root_label != -1) {
                    //     target_labels.push_back(opt.root_label);
                    //     std::cout << "[Info] No predicate labels, using ROOT_LABEL: " << opt.root_label << std::endl;
                    // } else {
                    //     std::cerr << "[Error] No labels specified for Oracle check!" << std::endl;
                    // }
                    if (opt.root_label != -1) {
                         // 这里同样如果是 raw root_label 也可能匹配不上
                         target_query_nodes.push_back(0); // 直接当第一个点
                    } else {
                        std::cerr << "[Error] No explicit nodes specified for Oracle check!" << std::endl;
                    }
                }


                if (run_fastesto_budget_curve) {
                    int oracle_cost = get_budget_limit_optimized(query_name);
                    if (oracle_cost <= 0) {
                        oracle_cost = 500; // 默认预算
                        std::cout << "[Info] FOIS_nrs base budget not found, will rely strictly on Curve CSV." << std::endl;
                    } else {
                        std::cout << "[Info] Oracle cost (baseline nrs): " << oracle_cost << std::endl;
                    }

                    double t_true = 0.0;
                    if (true_cnt.find(query_name) != true_cnt.end()) {
                        t_true = true_cnt[query_name];
                    }

                    for (int run_id = 1; run_id <= fastesto_runs; ++run_id) {
                        for (double frac : fastesto_budget_fracs) {
                            // int budget_n = std::max(1, (int)std::ceil(oracle_cost*1.2));
                            int oracle_cost = get_oracle_cost_from_curve(query_name, frac);
                            if (oracle_cost <= 0) {
                                // fallback：如果曲线里没有，退回旧缓存
                                oracle_cost = get_budget_limit_optimized(query_name);
                            }
                            if (oracle_cost <= 0) {
                                oracle_cost = 500; // 最后兜底
                            }
                            int budget_n = std::max(1, oracle_cost);
                            // est = estimator.EstimateWithMultiPredicate(P, target_labels, budget_n);
                            est = (agg_func == CardinalityEstimation::AGG_COUNT)
                                ? estimator.EstimateWithMultiPredicate(P, target_query_nodes, budget_n)
                                : estimator.EstimateWithMultiPredicateAgg(P, target_query_nodes, agg_func, sum_label, budget_n);
                            dict res_info = estimator.GetResult();
                            int unique_nodes = 0;
                            if (res_info.count("#UniqueOracleNodes")) {
                                unique_nodes = (int)std::any_cast<double>(res_info["#UniqueOracleNodes"]);
                            }

                            append_fastesto_budget_curve_row(
                                fastesto_budget_curve_path,
                                query_name,
                                run_id,
                                frac,
                                budget_n,
                                t_true,
                                est,
                                unique_nodes,
                                0,
                                oracle_cost,
                                "FastestO"
                            );

                            std::cout << "  [FastestO Curve] run=" << run_id
                                      << " frac=" << frac
                                      << " budget_n=" << budget_n
                                      << " est=" << est << std::endl;
                        }
                    }

                }
                else{
                int budget = -1;
                budget = get_budget_limit_optimized(query_name);
                if (budget != -1) budget = static_cast<int>(std::ceil(budget * 1.08));
                else budget = 500;

                if (budget != -1) {
                    std::cout << "[Info] Retrieved budget limit " << budget << " from " << summary_run1_path << std::endl;
                } else {
                    budget = 300; // 默认预算
                    std::cout << "[Info] No budget limit found in " << summary_run1_path << ". Using default budget." << budget << std::endl;
                }
                // 1. 获取估计值
                // 调用多谓词估计
                // est = estimator.EstimateWithMultiPredicate(P, target_labels, budget);
                est = (agg_func == CardinalityEstimation::AGG_COUNT)
                    ? estimator.EstimateWithMultiPredicate(P, target_query_nodes, budget)
                    : estimator.EstimateWithMultiPredicateAgg(P, target_query_nodes, agg_func, sum_label, budget);
                    
                // est = estimator.EstimateWithPredicate(P, opt.root_label, budget);
                // 2. 获取统计信息 (#UniqueOracleNodes)
                dict res_info = estimator.GetResult();
                int unique_nodes = 0;
                if (res_info.count("#UniqueOracleNodes")) {
                    unique_nodes = (int)std::any_cast<double>(res_info["#UniqueOracleNodes"]);
                }
                // 3. 【核心修改】 调用辅助函数保存 CSV
                // 保存 sampled_node_count
                save_sampled_node_count(sampled_node_count_path, query_name, unique_nodes);
                // 保存 results_summary (注意：这依赖于 Python 脚本已经生成了该文件并包含了此查询的其他方法记录)
                append_to_results_summary(results_summary_path, query_name, est, unique_nodes);
                std::cout << "  [Result] EstWithPredicate: " << est << std::endl;
                }
            }else { // --- 情况 2b: 普通的全局基数估计 (默认) ---
                std::cout << "\nStart Processing (Global) ws " << query_name << std::endl;
                est = estimator.EstimateEmbeddings(P, query_name, dataset);
                basic_estimates_for_json.push_back({query_name, est});
            }
            dict query_result = estimator.GetResult();
            query_result["Est"] = est;
            if (true_cnt.find(query_name)!= true_cnt.end()) {
                query_result["Truth"] = std::any(true_cnt[query_name]*1.0);
                query_result["logQError"] = std::any(logQError(true_cnt[query_name]*1.0, est));
            }
            for (auto &key : print_order) {
                if (query_result.find(key) == query_result.end()) continue;
                std::any value = query_result[key];
                if (double_type_results.find(key) != double_type_results.end())
                    fprintf(stdout, "  [Result] %-20s: %.04lf\n", key.c_str(), std::any_cast<double>(value));
                else if (scientific_type_results.find(key)!= scientific_type_results.end())
                    fprintf(stdout, "  [Result] %-20s: %.04g\n", key.c_str(), std::any_cast<double>(value));
                else if (longlong_type_results.find(key)!= longlong_type_results.end())
                    fprintf(stdout, "  [Result] %-20s: %lld\n", key.c_str(), std::any_cast<long long>(value));
                else
                    fprintf(stdout, "  [Result] %-20s: %d\n", key.c_str(), std::any_cast<int>(value));
            }
            cout << query_name << " Finished!\n";
            fflush(stdout);
            results.push_back(query_result);
        }
    }
    // [New] 将收集到的基础估计结果写入 JSON 文件
    if (!basic_estimates_for_json.empty()) {
        std::string json_path = results_dir + "basic_estimates.json";
        std::ofstream json_out(json_path);
        if (json_out.is_open()) {
            json_out << "{\n";
            for (size_t i = 0; i < basic_estimates_for_json.size(); ++i) {
                // 只保留文件名作为 key，去掉路径
                std::string key = get_filename_only(basic_estimates_for_json[i].first);
                double val = basic_estimates_for_json[i].second;
                
                json_out << "  \"" << key << "\": " << val;
                
                // 如果不是最后一个元素，添加逗号
                if (i < basic_estimates_for_json.size() - 1) {
                    json_out << ",";
                }
                json_out << "\n";
            }
            json_out << "}\n";
            json_out.close();
            std::cout << "[Info] Saved basic estimates to JSON: " << json_path << std::endl;
        } else {
            std::cerr << "[Error] Cannot open " << json_path << " for writing JSON." << std::endl;
        }
    }
}