#pragma once

#include "SubgraphCounting/Option.h"
#include "SubgraphMatching/CandidateSpace.h"
#include "SubgraphCounting/CandidateTreeSampling.h"
#include "SubgraphCounting/CandidateGraphSampling.h"
#include "SubgraphCounting/OracleManager.h"


// #include "SubgraphCounting/TreeRejectionSampling.h"
/**
 * @brief Subgraph Cardinality Estimation : Given G and P, approximate the number of embeddings of P in G.
 * @date 2023-05-01
 * @author Wonseok Shin
 * @ref
 */

namespace GraphLib
{
    using SubgraphMatching::DataGraph, SubgraphMatching::PatternGraph, SubgraphMatching::CandidateSpace;
    namespace CardinalityEstimation
    {
        class FaSTestCardinalityEstimation
        {
            CandidateSpace *CS;
            DataGraph *data_;
            PatternGraph *query_;
            CardEstOption opt_;
            dict result;
            CandidateTreeSampler *TS;
            CandidateGraphSampler *GS;
            OracleManager *oracle_mgr = nullptr;

        public:
            FaSTestCardinalityEstimation(DataGraph *data, CardEstOption opt)
            {
                query_ = nullptr; 
                data_ = data;
                opt_ = opt;

                CS = new CandidateSpace(data, opt);
                TS = new CandidateTreeSampler(data, opt);
                GS = new CandidateGraphSampler(data, opt);
                result.clear();
            };
            // 初始化 OracleManager
            OracleManager* GetOracleManager() const { return oracle_mgr; }
            void SetOracleManager(OracleManager* mgr) { oracle_mgr = mgr; }
            
            void InitOracle(const std::string &dataset_path)
            {
                oracle_mgr = new OracleManager();
                // data_ 是 DataGraph 指针
                oracle_mgr->Load(dataset_path, data_->GetNumVertices());
            }

            // [New] 初始化多源 Oracle (Post + Comment)
            // void InitMultiOracle(const std::string &dataset_path,
            //                      const std::string &post_col,
            //                      const std::string &comment_col)
            // {
            //     if (oracle_mgr == nullptr)
            //     {
            //         oracle_mgr = new OracleManager();
            //     }
            //     // 调用 OracleManager 的 LoadMulti
            //     oracle_mgr->LoadMulti(dataset_path, data_->GetNumVertices(), post_col, comment_col);
            // }

            void InitMultiOracle(const std::string& dataset_path,
                                const std::string& table1_name, const std::string& col1, 
                                const std::string& table2_name, const std::string& col2)
            {
                if (oracle_mgr == nullptr)
                {
                    oracle_mgr = new OracleManager();
                }
                // 调用 OracleManager 的 LoadMulti，传入新的表名和列名参数
                oracle_mgr->LoadMulti(dataset_path, data_->GetNumVertices(), table1_name, col1, table2_name, col2);
            }

            void InitAggColumn(const std::string &dataset_path,
                               const std::string &table,
                               const std::string &col)
            {
                if (oracle_mgr == nullptr)
                    oracle_mgr = new OracleManager();
                oracle_mgr->LoadAggColumn(dataset_path, data_->GetNumVertices(), table, col);
            }

            // 单谓词估计接口函数
            double EstimateWithPredicate(PatternGraph *query, int infer_label, int custom_budget = -1)
            {
                result.clear();
                query_ = query;

                // 1. 构建 CS
                CS->BuildCS(query_);

                // 2. 预处理树采样 (计算 total_trees_)
                TS->Preprocess(query_, CS);

                if (oracle_mgr == nullptr)
                {
                    std::cerr << "[Error] OracleManager not initialized!" << std::endl;
                    return 0.0;
                }

                // 3. 调用新采样函数
                double est = TS->EstimateWithOraclePredicate(infer_label, *oracle_mgr, custom_budget);

                // 4. 收集结果
                for (auto &[key, value] : TS->GetInfo())
                {
                    result[key] = value;
                }
                return est;
            }

            //  多谓词估计接口函数
            double EstimateWithMultiPredicate(PatternGraph *query,
                                              const std::vector<int> &target_labels,
                                              int custom_budget = -1)
            {
                result.clear();
                // query_ = query;

                // // 1. 构建 CS
                // CS->BuildCS(query_);

                // // 2. 预处理 (必须的，用于计算 total_trees_)
                // TS->Preprocess(query_, CS);
                if (query_ != query) {
                    query_ = query;
                    CS->BuildCS(query_);
                    TS->Preprocess(query_, CS);
                }

                if (oracle_mgr == nullptr)
                {
                    std::cerr << "[Error] OracleManager not initialized! Call InitMultiOracle first." << std::endl;
                    return 0.0;
                }

                // 3. 调用采样器的多谓词函数
                double est = TS->EstimateWithMultiOraclePredicate(target_labels, *oracle_mgr, custom_budget);

                // 4. 收集统计信息
                for (auto &[key, value] : TS->GetInfo())
                {
                    result[key] = value;
                }
                return est;
            }

            double EstimateWithMultiPredicateAgg(PatternGraph *query,
                                                 const std::vector<int> &target_labels,
                                                 AggFunc agg_func,
                                                 int agg_sum_label,
                                                 int custom_budget = -1)
            {
                result.clear();
                
                // query_ = query;
                // CS->BuildCS(query_);
                // TS->Preprocess(query_, CS);

                if (query_ != query) {
                    query_ = query;
                    CS->BuildCS(query_);
                    TS->Preprocess(query_, CS);
                }

                if (oracle_mgr == nullptr)
                {
                    std::cerr << "[Error] OracleManager not initialized!\n";
                    return 0.0;
                }

                double est = TS->EstimateWithMultiOraclePredicate(target_labels, *oracle_mgr,
                                                                  custom_budget, agg_func, agg_sum_label);
                for (auto &[key, value] : TS->GetInfo())
                    result[key] = value;
                return est;
            }

            // std::pair<double, std::map<std::vector<int>, double>>
            // EstimateCoreInstancesAgg(PatternGraph *query,
            //                         const std::vector<int> &core_labels,
            //                         AggFunc agg_func,
            //                         int agg_sum_label)
            std::pair<double, std::map<std::vector<int>, double>> 
            EstimateCoreInstancesAgg(PatternGraph *query, 
                                    const std::vector<int> &core_query_nodes,
                                    AggFunc agg_func, int agg_sum_label)
            {
                result.clear();
                query_ = query;

                CS->BuildCS(query_);
                TS->Preprocess(query_, CS);

                if (agg_func == AGG_SUM && oracle_mgr == nullptr)
                {
                    std::cerr << "[Error] SUM requires InitAggColumn() first.\n";
                    return {0.0, {}};
                }

                // auto ret = TS->EstimateCoreInstanceFrequency(core_labels, oracle_mgr, agg_func, agg_sum_label);
                auto ret = TS->EstimateCoreInstanceFrequency(core_query_nodes, oracle_mgr, agg_func, agg_sum_label);
                for (auto &[key, value] : TS->GetInfo())
                    result[key] = value;
                return ret;
            }

            dict GetResult() { return result; }
            double EstimateEmbeddings(PatternGraph *query, const std::string &query_name, const std::string &dataset_name)
            {
                result.clear();
                double query_time = 0.0;
                query_ = query;

                // === Step 1. 构建候选空间 ===
                CS->BuildCS(query_);
                for (auto &[key, value] : CS->GetCSInfo())
                {
                    result[key] = value;
                }

                // === Step 2. 预处理树采样器 ===
                TS->Preprocess(query, CS);

                // === Step 3. 执行带节点频率统计的采样 ===EstimateWithNodeFrequency
                // auto ts_result = TS->EstimateWithNodeFrequency();  // ⭐ 改这里
                // auto ts_result = TS->EstimateWithNodeFrequencyByLabel(opt_.root_label); // ⭐ 改这里
                // 在 Preprocess() 之后：
                auto ts_result = TS->EstimateWithNodeFrequency(); // Tq.root 已由 BuildSpanningTree 按 opt 固定
                double est = ts_result.first;                     // 总体估计值
                auto &node_est = ts_result.second;                // 每个 root 节点的频率估计

                // === Step 4. 保存采样器的统计信息 ===
                for (auto &[key, value] : TS->GetInfo())
                {
                    result[key] = value;
                }

                result["GraphSampleTime"] = 0.0;
                // === Step 6. 如果树采样太稀疏，切换图采样 ===
                int success = (int)std::any_cast<int>(result["#TreeSuccess"]);
                if (success <= 10)
                {
                    std::cout << "[WS Graph] Few successes, fallback to graph sampling.\n";
                    std::ofstream log_file("/home/wangshuo/projects/Fastest_source_code/dataset/yeast/graph_sample.txt", std::ios::app);
                    if (log_file.is_open())
                    {
                        log_file << query_name << std::endl;
                        log_file.close();
                    }
                    GS->Preprocess(query, CS);
                    est = GS->Estimate(ceil((double)(opt_.ub_initial * query_->GetNumVertices()) / sqrt(success + 1)));

                    for (auto &[key, value] : GS->GetInfo())
                    {
                        result[key] = value;
                    }
                }
                // === Step 5. 输出每个节点的估计值 ===
                // std::string sv_out_path = "/home/wangshuo/projects/FaSTest-main/dataset/sv/sv_estimate_result.txt";
                std::string sv_out_path = "/home/wangshuo/resource/datasets/parler_data/" + dataset_name + "/results/in_estimateW_result.txt";
                std::cout << "[INFO] sv_out_path " << sv_out_path << std::endl;
                std::ofstream fout(sv_out_path, std::ios::app);
                if (fout.is_open())
                {
                    fout << "Query: " << query_name << "\n";
                    fout << "All Est: " << est << "\n";
                    for (auto &[node_id, est_val] : node_est)
                    {
                        fout << node_id << " " << est_val << "\n";
                    }
                    fout << "----------------------------------------\n";
                    fout.close();
                    std::cout << "[Info] Node-level estimates written to " << sv_out_path << std::endl;
                }
                // === Step 7. 汇总时间统计 ===
                query_time = std::any_cast<double>(result["CSBuildTime"]) + std::any_cast<double>(result["TreeCountTime"]) + std::any_cast<double>(result["TreeSampleTime"]) + std::any_cast<double>(result["GraphSampleTime"]);

                result["QueryTime"] = query_time;
                std::cout << "[INFO] QueryTime for " << query_name << " = "
                          << std::fixed << std::setprecision(4)
                          << query_time << " ms" << std::endl;

                return est;
            }

            // std::pair<double, std::map<std::vector<int>, double>> EstimateCoreInstances(PatternGraph *query, const std::vector<int> &core_labels)
            std::pair<double, std::map<std::vector<int>, double>> EstimateCoreInstances(PatternGraph *query, const std::vector<int> &core_query_nodes)
            {
                result.clear();
                query_ = query;

                // 1. 构建候选空间
                CS->BuildCS(query_);

                // 2. 预处理树采样器，这一步会计算 total_trees_，是估算所必需的
                TS->Preprocess(query_, CS);

                // 3. 检查树采样成功率，如果过低，新方法可能不准确 (可选的健壮性检查)
                int success_check = 0;
                if (TS->GetInfo().count("#TreeSuccess"))
                {
                    success_check = std::any_cast<int>(TS->GetInfo().at("#TreeSuccess"));
                }
                if (success_check <= 10 && TS->GetInfo().count("#TreeTrials") && std::any_cast<int>(TS->GetInfo().at("#TreeTrials")) >= 50000)
                {
                    std::cerr << "[Warning] Tree sampling success rate is very low. Core instance estimates may be inaccurate." << std::endl;
                }

                // 4. 调用核心实现
                // auto instance_results_pair = TS->EstimateCoreInstanceFrequency(core_labels);
                auto instance_results_pair = TS->EstimateCoreInstanceFrequency(core_query_nodes);

                // 5. 收集统计信息
                for (auto &[key, value] : TS->GetInfo())
                {
                    result[key] = value;
                }

                return instance_results_pair;
            };
            /**
             * @brief Estimate, for a given query and node set sv, how many embeddings include each node.
             */
            std::unordered_map<int, double> EstimateContainingNodes(PatternGraph *query, const std::vector<int> &sv)
            {
                result.clear();
                query_ = query;
                CS->BuildCS(query_);
                for (auto &[key, value] : CS->GetCSInfo())
                {
                    result[key] = value;
                }

                // Step 1: 候选树采样预处理
                TS->Preprocess(query_, CS);

                // Step 2: 调用我们实现的自适应采样函数
                auto res = TS->EstimateContainingNode(sv);

                // Step 3: 保存统计信息
                for (auto &[key, value] : TS->GetInfo())
                {
                    result[key] = value;
                }

                // Step 4: 返回结果
                return res;
            };
        };
    }
}