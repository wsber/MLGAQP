#pragma once
#include <boost/math/distributions.hpp>
#include "SubgraphMatching/CandidateSpace.h"
#include "SubgraphCounting/OracleManager.h"
#include <unordered_set> 

static std::random_device rd;
static std::mt19937 gen(rd());
inline int sample_from_distribution(std::discrete_distribution<int> &weighted_distr) {
    return weighted_distr(gen);
}

namespace GraphLib {
    using SubgraphMatching::DataGraph, SubgraphMatching::PatternGraph;
    using SubgraphMatching::CandidateSpace;
    using std::vector;
    namespace CardinalityEstimation {
        struct QueryTree {
            PatternGraph *query_;
            vector<vector<int>> tree_adj_list, tree_children;
            vector<int> tree_sequence;
            vector<int> parent, child_index;
            int root;

            void Initialize(PatternGraph *query, int root_idx) {
                query_ = query;
                tree_adj_list.clear();
                tree_children.clear();
                tree_sequence.clear();
                parent.clear();
                child_index.clear();
                root = root_idx;

                tree_adj_list.resize(query_->GetNumVertices());
                tree_children.resize(query_->GetNumVertices());
                tree_sequence.resize(query_->GetNumVertices(), -1);
                parent.resize(query_->GetNumVertices(), -1);
                child_index.resize(query_->GetNumVertices(), -1);
            }

            void AddEdge(int u, int v) {
                tree_adj_list[u].push_back(v);
                tree_adj_list[v].push_back(u);
            }

            void BuildTree() {
                bool *visit = new bool[query_->GetNumVertices()];
                std::fill(visit, visit + query_->GetNumVertices(), false);
                std::queue<int> q;
                int id = 0;
                parent[root] = -1;
                tree_sequence[id++] = root;
                q.push(root);
                visit[root] = true;
                while (!q.empty()) {
                    int v = q.front();
                    q.pop();
                    for (int c : tree_adj_list[v]) {
                        if (!visit[c]) {
                            q.push(c);
                            visit[c] = true;
                            child_index[c] = tree_children[v].size();
                            tree_children[v].push_back(c);
                            parent[c] = v;
                            tree_sequence[id++] = c;
                        }
                    }
                }
                delete[] visit;
            }

            vector<int>& GetChildren(int v) {return tree_children[v];}
            int GetParent(int v) {return parent[v];}
            int GetChildIndex(int v) {return child_index[v];}
            int GetKthVertex(int k) {return tree_sequence[k];}
        };
        class CandidateTreeSampler {
        protected:
            dict info;
            DataGraph *data_;
            PatternGraph *query_;
            CandidateSpace *CS;
            CardEstOption opt;

            double **num_trees_, total_trees_;
            bool *seen;
            // vector<vector<vector<vector<int>>>> sample_candidates_;
            vector<vector<vector<vector<double>>>> sample_candidate_weights_;
            vector<vector<vector<std::discrete_distribution<int>>>> sample_dist_;
            std::vector<double> root_sample_dist_;
            vector<int> root_candidates_, sample;
            vector<double> root_weights_;
            std::discrete_distribution<int> sample_root_dist_;

            QueryTree Tq;
        public:
            dict GetInfo() {return info;}
            CandidateTreeSampler(DataGraph *data, CardEstOption option) {
                data_ = data;
                opt = option;
                num_trees_ = new double *[opt.MAX_QUERY_VERTEX];
                for (int i = 0; i < opt.MAX_QUERY_VERTEX; i++) {
                    num_trees_[i] = new double[data_->GetNumVertices()];
                }
                seen = new bool[data_->GetNumVertices()]();
            }


            void Preprocess(PatternGraph *query, CandidateSpace *cs) {
                info.clear();
                Timer timer; timer.Start();
                query_ = query;
                CS = cs;
                sample_dist_.clear();
                // sample_candidates_.clear();
                sample_candidate_weights_.clear();
                root_candidates_.clear();

                sample.resize(query_->GetNumVertices(), -1);
                std::memset(seen, 0, sizeof(bool) * data_->GetNumVertices());
                BuildSpanningTree();
                CountCandidateTrees();
                timer.Stop();
                info["TreeCountTime"] = timer.GetTime();
                info["#CandTree"] = total_trees_;
            };

            void BuildSpanningTree() {
                // If random strategy requested, keep random behavior (no change)
                if (opt.treegen_strategy == CardinalityEstimation::TREEGEN_RANDOM) {
                    int root_node = rand() % query_->GetNumVertices();
                    Tq.Initialize(query_, root_node);
                    int num_discovered = 1;
                    std::vector<int> is_discovered(query_->GetNumVertices(), false);
                    is_discovered[root_node] = true;
                    int cur = root_node;
                    while (num_discovered < query_->GetNumVertices()) {
                        int d = query_->GetDegree(cur);
                        int di = rand() % d;
                        int v = query_->GetNeighbors(cur)[di];
                        if (!is_discovered[v]) {
                            Tq.AddEdge(cur, v);
                            is_discovered[v] = true;
                            num_discovered++;
                        }
                        cur = v;
                    }
                    Tq.BuildTree();
                    return;
                }

                // ---------- Begin modified selection of root ----------
                int qV = query_->GetNumVertices();

                // default candidate: pick query vertex with smallest candidate set (existing logic)
                int default_root = 0;
                int num_root_cands = CS->GetCandidateSetSize(0);
                for (int i = 0; i < qV; ++i) {
                    if (CS->GetCandidateSetSize(i) < num_root_cands) {
                        num_root_cands = CS->GetCandidateSetSize(i);
                        default_root = i;
                    }
                }

                int chosen_root = default_root;

                // 1) If user specified a query index, use it (if valid)
                if (opt.root_query_index >= 0 && opt.root_query_index < qV) {
                    chosen_root = opt.root_query_index;
                    std::cerr << "[Info] Using user-specified query root index: " << chosen_root << std::endl;
                }
                // 2) Else if user specified a label, find the first query vertex with that label
                else if (opt.root_label >= 0) {
                    int found = -1;
                    for (int i = 0; i < qV; ++i) {
                        if (query_->GetVertexLabel(i) == opt.root_label) {
                            found = i;
                            break;
                        }
                    }
                    if (found != -1) {
                        chosen_root = found;
                        std::cerr << "[Info] Using user-specified root label " << opt.root_label
                                << " -> query node " << chosen_root << std::endl;
                    } else {
                        std::cerr << "[Warning] Requested root label " << opt.root_label
                                << " not found in query. Falling back to default root " << default_root << std::endl;
                    }
                }
                else {// 否则：没有用户指定根，使用系统默认（已在 default_root 计算完成）
                    // 这是你要求的：在**用户不指定标签**时，打印自动选定的根及其标签
                    chosen_root = default_root;
                    int root_label = query_->GetVertexLabel(chosen_root);
                    // 说明：query_->GetVertexLabel 返回的是内部的 transferred label（由 DataGraph::TransformLabel 设置）
                    std::cout << "[Info] Auto-selected query root index " << chosen_root
                            << " with (transferred) label " << root_label << std::endl;
                }
                            // ---------- End modified selection of root ----------

                // Now build a spanning tree with chosen_root as root (keep original MST-style logic)
                Tq.Initialize(query_, chosen_root);
                // build edges using density / MST code (the original code that builds edges)
                std::vector<std::pair<double, std::pair<int, int>>> edges;
                for (int i = 0; i < qV; i++) {
                    for (int q_neighbor : query_->GetNeighbors(i)) {
                        if (i > q_neighbor) continue;
                        double density = 0.0;
                        for (int cand_idx = 0; cand_idx < CS->GetCandidateSetSize(i); cand_idx++) {
                            int num_cs_neighbor = CS->GetCandidateNeighbors(i, cand_idx, q_neighbor).size();
                            density += num_cs_neighbor;
                        }
                        if (opt.treegen_strategy == CardinalityEstimation::TREEGEN_DENSITY_MST) {
                            density /= ((CS->GetCandidateSetSize(i) * 1.0) * (CS->GetCandidateSetSize(q_neighbor) * 1.0));
                        }
                        if (density > 0) {
                            edges.push_back({density * 1.0, {i, q_neighbor}});
                        }
                    }
                }
                std::sort(edges.begin(), edges.end());
                int num_tree_edges = 0;
                std::vector<int> deg(qV, 0);
                UnionFind uf(qV);
                while (num_tree_edges + 1 < qV) {
                    double minw = 1e9;
                    std::pair<int, int> me = {-1,-1};
                    for (auto &pr : edges) {
                        double w = pr.first;
                        auto e = pr.second;
                        auto [u, v] = e;
                        if (uf.find(u) == uf.find(v)) continue;
                        if (minw > w + deg[u] * 1e-7) {
                            minw = w + deg[u] * 1e-7;
                            me = e;
                        }
                    }
                    if (me.first == -1) break; // defensive
                    uf.unite(me.first, me.second);
                    deg[me.first]++; deg[me.second]++;
                    Tq.AddEdge(me.first, me.second);
                    num_tree_edges++;
                }
                Tq.BuildTree();
            }


            void CountCandidateTrees() {
                for (int i = 0; i < query_->GetNumVertices(); i++) {
                    memset(num_trees_[i], 0, sizeof(double) * CS->GetCandidateSetSize(i));
                }
                int cnt = 0;
                sample_candidate_weights_.resize(query_->GetNumVertices());
                sample_dist_.resize(query_->GetNumVertices());
                for (int i = 0; i < query_->GetNumVertices(); i++) {
                    int u = Tq.GetKthVertex(query_->GetNumVertices() - i - 1);
                    int num_cands = CS->GetCandidateSetSize(u);
                    auto children = Tq.GetChildren(u);
                    int num_children = children.size();
                    sample_candidate_weights_[u].resize(num_cands);
                    sample_dist_[u].resize(num_cands);

                    std::vector<double> tmp_num_child(num_children);
                    for (int cs_idx = 0; cs_idx < num_cands; cs_idx++) {
                        sample_candidate_weights_[u][cs_idx].resize(num_children);
                        sample_dist_[u][cs_idx].resize(num_children);

                        double num_ = 1.0;
                        std::fill(tmp_num_child.begin(), tmp_num_child.end(), 0.0);
                        for (int uc_idx = 0; uc_idx < num_children; uc_idx++) {
                            int uc = children[uc_idx];
                            auto candidate_neighbors = CS->GetCandidateNeighbors(u, cs_idx, uc);
                            sample_candidate_weights_[u][cs_idx][uc_idx].resize(candidate_neighbors.size());
                            for (int j = 0; j < candidate_neighbors.size(); j++) {
                                int vc_idx = candidate_neighbors[j];
                                tmp_num_child[uc_idx] += num_trees_[uc][vc_idx];
                                sample_candidate_weights_[u][cs_idx][uc_idx][j] = num_trees_[uc][vc_idx];
                            }
                        }
                        for (int j = 0; j < num_children; j++) {
                            num_ *= tmp_num_child[j];
                            sample_dist_[u][cs_idx][j] = std::discrete_distribution<int>(
                                    sample_candidate_weights_[u][cs_idx][j].begin(),
                                    sample_candidate_weights_[u][cs_idx][j].end());
                            cnt++;
                        }
                        num_trees_[u][cs_idx] = num_;
                    }
                }

                total_trees_ = 0.0;
                root_candidates_.clear();
                root_weights_.clear();
                int root = Tq.root;
                int root_candidate_size = CS->GetCandidateSetSize(root);
                for (int root_candidate_idx = 0; root_candidate_idx < root_candidate_size; ++root_candidate_idx) {
                    total_trees_ += num_trees_[root][root_candidate_idx];
                    if (num_trees_[root][root_candidate_idx] > 0) {
                        root_candidates_.emplace_back(root_candidate_idx);
                        root_weights_.emplace_back(num_trees_[root][root_candidate_idx]);
                    }
                }
                sample_root_dist_ = std::discrete_distribution<int>(root_weights_.begin(), root_weights_.end());
                // ⭐ 新增：初始化 root_sample_dist_（归一化形式）
                root_sample_dist_.resize(root_candidate_size);
                double total_w = 0.0;
                for (int i = 0; i < root_candidate_size; ++i) total_w += num_trees_[root][i];
                if (total_w == 0) total_w = 1;
                for (int i = 0; i < root_candidate_size; ++i)
                    root_sample_dist_[i] = num_trees_[root][i] / total_w;
            };

            bool GetSampleTree(int fixed_root_idx = -1) {
                std::fill(sample.begin(), sample.end(), -1);
                bool valid = true;

                // 若给定 root，则固定之，否则随机选一个
                int root_idx = (fixed_root_idx >= 0)
                    ? fixed_root_idx
                    : root_candidates_[sample_from_distribution(sample_root_dist_)];
                sample[Tq.root] = root_idx;
                seen[CS->GetCandidate(Tq.root, sample[Tq.root])] = true;

                for (int i = 0; i < query_->GetNumVertices(); ++i) {
                    int u = Tq.GetKthVertex(i);
                    int v_idx = sample[u];
                    auto &children = Tq.GetChildren(u);
                    for (int uc_idx = 0; uc_idx < children.size(); ++uc_idx) {
                        int uc = children[uc_idx];
                        int vc_idx = sample_from_distribution(sample_dist_[u][v_idx][uc_idx]);
                        sample[uc] = CS->GetCandidateNeighbor(u, v_idx, uc, vc_idx);
                        int cand = CS->GetCandidate(uc, sample[uc]);
                        if (seen[cand]) { valid = false; goto CLEANUP; }
                        seen[cand] = true;
                    }
                }

            CLEANUP:
                for (int i = 0; i < query_->GetNumVertices(); i++) {
                    if (sample[i] >= 0) {
                        int cand = CS->GetCandidate(i, sample[i]);
                        seen[cand] = false;
                    }
                }
                if (!valid) return false;

                for (int i = 0; i < query_->GetNumVertices(); i++) {
                    if (sample[i] == -1) return false;
                    sample[i] = CS->GetCandidate(i, sample[i]);
                }
                for (int i = 0; i < query_->GetNumVertices(); i++) {
                    for (int qe : query_->GetAllIncidentEdges(i)) {
                        int j = query_->GetOppositePoint(qe);
                        int de = data_->GetEdgeIndex(sample[i], sample[j]);
                        if (de == -1) return false;
                        if (data_->GetEdgeLabel(de) != query_->GetEdgeLabel(qe)) return false;
                    }
                }
                return true;
            }

            std::pair<double, int> Estimate() {
                Timer timer; timer.Start();
                int success = 0, trials = 0;
                while (++trials) {
                    auto result = GetSampleTree();
                    if (result) success++;
                    double rhohat = (success * 1.0 / trials);
                    if (trials == 50000 and success <= 10) {
                        timer.Stop();
                        info["#TreeTrials"] = trials;
                        info["#TreeSuccess"] = success;
                        info["TreeSampleTime"] = timer.GetTime();
                        return {-1, success};
                    }
                    if (trials >= 1000 and trials % 100 == 0) {
                        long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05/2);
                        long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05/2);
                        if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25) {
                            timer.Stop();
                            break;
                        }
                    }
                }
                auto est = std::make_pair((success * 1.0 / (trials * 1.0)) * total_trees_, success);
                info["#TreeTrials"] = trials;
                info["#TreeSuccess"] = success;
                info["TreeSampleTime"] = timer.GetTime();
                return est;
            }
            
            std::pair<double, std::unordered_map<int, double>> EstimateWithNodeFrequency() {
                Timer timer; 
                timer.Start();

                int success = 0, trials = 0;
                int cand_size = CS->GetCandidateSetSize(Tq.root);
                int current_budget = opt.sample_budget; 

                // 每个 root candidate 成功的次数
                std::vector<int> success_per_cand(cand_size, 0);

                // === 主采样循环 ===
                while (++trials) {
                    // 1️⃣ 从 root 候选集合中按概率分布采样
                    std::discrete_distribution<int> tmp_dist(root_sample_dist_.begin(), root_sample_dist_.end());
                    int root_idx = tmp_dist(gen);
                    int root_node = CS->GetCandidate(Tq.root, root_idx);

                    // 2️⃣ 在该 root 下采样一棵候选树
                    bool result = GetSampleTree(root_idx);

                    if (result) {
                        success++;
                        success_per_cand[root_idx]++; // 成功时计数
                    }

                    // 3️⃣ 自适应停止条件（原逻辑保留）
                    double rhohat = (success * 1.0 / trials);
                    if (trials == 50000 && success <= 10) break;
                    if (trials >= current_budget && trials % 100 == 0) {
                        long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05/2);
                        long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05/2);
                        if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25)
                            break;
                    }
                }

                // === Step 1. 全局估计值 ===
                double est_total = (success * 1.0 / trials) * total_trees_;

                // === Step 2. 每个 root candidate 的无偏节点频率估计 ===
                std::unordered_map<int, double> node_est;
                double sum_node_est = 0.0;

                for (int i = 0; i < cand_size; ++i) {
                    int node_id = CS->GetCandidate(Tq.root, i);
                    double num_trees_i = num_trees_[Tq.root][i];
                    if (num_trees_i <= 0) continue;

                    // 每个 root candidate 被选中的概率 p_i
                    double p_i = num_trees_i / total_trees_;
                    if (p_i <= 0) continue;

                    // 观测到的成功率（含采样概率）
                    double obs_rate = success_per_cand[i] * 1.0 / trials;

                    // 无偏成功率 s_i = obs_rate / p_i
                    double s_i = obs_rate / p_i;

                    // 节点频率估计 f(v) = s_i × num_trees_i
                    double est_v = s_i * num_trees_i;
                    node_est[node_id] = est_v;

                    sum_node_est += est_v;
                }

                // === Step 3. 一致性检查（可选） ===
                std::cout << "[Check] Sum(node_est) = " << sum_node_est
                        << ", est_total = " << est_total
                        << ", ratio = " << (sum_node_est / (est_total + 1e-9)) << std::endl;

                // === Step 4. 时间与统计信息记录 ===
                timer.Stop();
                info["#TreeTrials"] = trials;
                info["#TreeSuccess"] = success;
                info["TreeSampleTime"] = timer.GetTime();

                // === Step 5. 返回结果 ===
                return {est_total, node_est};
            }
            
            std::pair<double, std::map<std::vector<int>, double>> EstimateCoreInstanceFrequency(const std::vector<int>& core_query_nodes_input,OracleManager* oracle_manager = nullptr,
            AggFunc agg_func = AGG_COUNT,
            int agg_sum_label = -1) {
                Timer timer;
                timer.Start();

                std::map<std::vector<int>, double> instance_estimates;

                std::vector<int> core_query_nodes = core_query_nodes_input;
                std::sort(core_query_nodes.begin(), core_query_nodes.end());
                core_query_nodes.erase(std::unique(core_query_nodes.begin(), core_query_nodes.end()), core_query_nodes.end());

                if (core_query_nodes.empty()) {
                    std::cerr << "[Warning] Empty core query node list." << std::endl;
                    return {0.0, instance_estimates};
                }

                for (int qid : core_query_nodes) {
                    if (qid < 0 || qid >= query_->GetNumVertices()) {
                        std::cerr << "[Warning] Invalid core query node id: " << qid << std::endl;
                        return {0.0, instance_estimates};
                    }
                }

                std::vector<int> agg_query_nodes;
                if (agg_func == AGG_SUM) {
                    if (oracle_manager == nullptr) {
                        std::cerr << "[Error] SUM requires OracleManager (agg column not loaded)\n";
                        return {0.0, instance_estimates};
                    }
                    
                    int internal_sum_label = data_->GetTransferredLabel(agg_sum_label);

                    for (int i = 0; i < query_->GetNumVertices(); ++i) {
                        if (query_->GetVertexLabel(i) == internal_sum_label) {
                            agg_query_nodes.push_back(i);
                        }
                    }
                    if (agg_query_nodes.empty()) {
                        std::cerr << "[Error] SUM label not found in query: " << agg_sum_label << std::endl;
                        return {0.0, instance_estimates};
                    }
                }

                int success = 0, trials = 0;
                std::map<std::vector<int>, double> agg_per_instance;
                double agg_total = 0.0;
                int current_budget = opt.sample_budget;

                std::cout << "[Check] Sample budget = " << current_budget << std::endl;

                while (++trials) {
                    bool result = GetSampleTree();

                    if (result) {
                        success++;

                        std::vector<int> instance_key;
                        instance_key.reserve(core_query_nodes.size());
                        for (int q_node : core_query_nodes) {
                            instance_key.push_back(sample[q_node]);
                        }

                        std::sort(instance_key.begin(), instance_key.end());

                        double g = 1.0;
                        if (agg_func == AGG_SUM) {
                            g = 0.0;
                            for (int node : agg_query_nodes) {
                                g += oracle_manager->GetAggValue(sample[node]);
                            }
                        }

                        agg_total += g;
                        agg_per_instance[instance_key] += g;
                    }

                    double rhohat = (success * 1.0 / trials);
                    if (trials == 50000 && success <= 10) break;
                    if (trials >= current_budget && trials % 100 == 0) {
                        long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05 / 2);
                        long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05 / 2);
                        if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25) {
                            break;
                        }
                    }
                }

                if (trials == 0 || success == 0) {
                    timer.Stop();
                    info["#TreeTrials"] = trials;
                    info["#TreeSuccess"] = success;
                    info["TreeSampleTime"] = timer.GetTime();
                    return {0.0, instance_estimates};
                }

                double est_total = (agg_total / trials) * total_trees_;

                double sum_instance_est = 0.0;
                for (auto const& [instance, agg_sum] : agg_per_instance) {
                    double est_instance = (agg_sum / trials) * total_trees_;
                    instance_estimates[instance] = est_instance;
                    sum_instance_est += est_instance;
                }

                std::cout << "[Check] Sum of instance estimates = " << sum_instance_est
                        << ", Global estimate = " << est_total
                        << ", Ratio = " << (sum_instance_est / (est_total + 1e-9)) << std::endl;

                timer.Stop();
                info["#TreeTrials"] = trials;
                info["#TreeSuccess"] = success;
                info["TreeSampleTime"] = timer.GetTime();

                return {est_total, instance_estimates};
            }

            double EstimateWithOraclePredicate(int infer_label,
                 OracleManager& oracle_manager, 
                 int custom_oracle_budget = -1,
                 AggFunc agg_func = AGG_COUNT,
                 int agg_sum_label = -1) {
                Timer timer; timer.Start();
                
                // 0) 若做 SUM：定位“要被 SUM 的查询变量”（语义B：只取一个槽位）
                int agg_query_node = -1;
                if (agg_func == AGG_SUM) {
                    if (agg_sum_label < 0) {
                        std::cerr << "[Error] AGG_SUM requires a valid agg_sum_label.\n";
                        return 0.0;
                    }

                    int internal_sum_label = data_->GetTransferredLabel(agg_sum_label);

                    for (int i = 0; i < query_->GetNumVertices(); ++i) {
                        if (query_->GetVertexLabel(i) == internal_sum_label) {
                            agg_query_node = i;
                            break;
                        }
                    }
                    if (agg_query_node == -1) {
                        std::cerr << "[Error] SUM label not found in query: " << agg_sum_label << std::endl;
                        return 0.0;
                    }
                }

                // 1) 定位“需要做单谓词 Oracle 检查”的查询变量（由 infer_label 指定）
                int target_query_node = -1;

                int internal_infer_label = data_->GetTransferredLabel(infer_label);

                for (int i = 0; i < query_->GetNumVertices(); ++i) {
                    if (query_->GetVertexLabel(i) == internal_infer_label) {
                        target_query_node = i;
                        break; 
                    }
                }

                if (target_query_node == -1) {
                    std::cerr << "[Error] Query does not contain node with infer label: " << infer_label << std::endl;
                    return 0.0;
                }

                int success = 0, trials = 0;
                // 原有的采样上限/预算（这里指Trials次数）
                double agg_success_sum = 0.0; // COUNT: 成功+1; SUM: 成功+value

                int sample_budget = opt.sample_budget; 
                std::unordered_set<int> unique_oracle_checked_nodes;
                
                bool use_custom_budget = (custom_oracle_budget != -1);
                std::cout << "[Info] Oracle budget mode: "
                          << (use_custom_budget ? "custom" : "default")
                          << " oracle budget: "
                          << (use_custom_budget ? std::to_string(custom_oracle_budget) : std::to_string(sample_budget))
                          << std::endl;
                while (true) {
                    // 执行结构采样
                    bool struct_valid = GetSampleTree(); 

                    if (!struct_valid) {
                         trials++;
                         // 【修正点1】：无论是否有 Oracle 预算限制，原有的采样上限检查都必须生效
                         if (trials == 50000 && success <= 10) break;
                         if (trials >= sample_budget && trials % 100 == 0) {
                              double rhohat = (success * 1.0 / trials);
                              long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05/2);
                              long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05/2);
                              if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25)
                                  break;
                         }
                         continue;
                    }

                    // 结构有效，准备检查 Oracle
                    int mapped_data_node = sample[target_query_node];
                    
                    if (use_custom_budget) {
                        // 检查是否是新节点
                        bool is_new = unique_oracle_checked_nodes.find(mapped_data_node) == unique_oracle_checked_nodes.end();
                        if (is_new) {
                            // 是新节点，需要消耗预算
                            if (unique_oracle_checked_nodes.size() >= custom_oracle_budget) {
                                // 【新限制】：预算耗尽，立即停止
                                break;
                            }
                            unique_oracle_checked_nodes.insert(mapped_data_node);
                        }
                        // 否则是旧节点，命中缓存（免费），继续执行
                    } else {
                         unique_oracle_checked_nodes.insert(mapped_data_node);
                    }

                    trials++; 

                    if (oracle_manager.CheckOracle(mapped_data_node, 0.5)) {
                        success++;
                                    // 成功样本的 g(sample)
                        double g = 1.0;
                        if (agg_func == AGG_SUM) {
                            // sample[agg_query_node] 是数据图 internal_id
                            g = oracle_manager.GetAggValue(sample[agg_query_node]);
                        }
                        agg_success_sum += g;
                    }

                    // 【修正点2】：同样，在成功采样后也要检查原有的停止条件
                    double rhohat = (success * 1.0 / trials);
                    if (trials == 50000 && success <= 10) break;
                    if (trials >= sample_budget && trials % 100 == 0) {
                        long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05/2);
                        long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05/2);
                        if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25)
                            break;
                    }
                }
                // double est = (trials > 0) ? (success * 1.0 / trials) * total_trees_ : 0.0;
                double est = (trials > 0) ? (agg_success_sum / trials) * total_trees_ : 0.0;

                timer.Stop();
                info["QueryTime"] = timer.GetTime(); 
                info["#TreeTrials"] = trials;
                info["#TreeSuccess"] = success; 
                info["#UniqueOracleNodes"] = (double)unique_oracle_checked_nodes.size();
                return est;
            }
            
            double EstimateWithMultiOraclePredicate(
                // const std::vector<int>& target_labels,
                const std::vector<int>& target_query_nodes_input,
                                        OracleManager& oracle_manager,
                                        int custom_oracle_budget = -1,
                                        AggFunc agg_func = AGG_COUNT,
                                        int agg_sum_label = -1) {
                Timer timer;
                timer.Start();

                std::vector<int> agg_query_nodes;
                if (agg_func == AGG_SUM) {
                    int internal_sum_label = data_->GetTransferredLabel(agg_sum_label);

                    for (int i = 0; i < query_->GetNumVertices(); ++i) {
                        if (query_->GetVertexLabel(i) == internal_sum_label) {
                            agg_query_nodes.push_back(i);
                        }
                    }
                    if (agg_query_nodes.empty()) {
                        std::cerr << "[Error] SUM label not found in query: " << agg_sum_label << std::endl;
                        return 0.0;
                    }
                }

                // 1) 按 label 收集查询点（与当前主流程保持一致）
                std::vector<int> target_query_nodes =target_query_nodes_input;

                if (target_query_nodes.empty()) {
                    std::cerr << "[Warning] No query nodes match target constraints." << std::endl;
                    return 0.0;
                }

                // 排序确保稳定
                std::sort(target_query_nodes.begin(), target_query_nodes.end());
                // 去重
                target_query_nodes.erase(std::unique(target_query_nodes.begin(), target_query_nodes.end()), target_query_nodes.end());

                int success = 0, trials = 0;
                int sample_budget = opt.sample_budget;
                bool use_custom_budget = (custom_oracle_budget != -1);

                // 预算去重集合：只对“新数据点首次检查”扣预算
                std::unordered_set<int> budget_unique_nodes;

                // Oracle结果缓存：旧点直接复用结果，不再重复调用Oracle
                std::unordered_map<int, bool> oracle_result_cache;

                // 统计信息
                long long oracle_calls = 0;      // 实际调用 CheckOracle 的次数
                long long oracle_cache_hits = 0; // 命中缓存次数
                long long oracle_budget_used = 0; // 实际消耗的预算次数（新点首次检查次数）

                double agg_success_sum = 0.0;

                while (true) {
                    bool struct_valid = GetSampleTree();

                    if (!struct_valid) {
                        trials++;
                        if (trials == 50000 && success <= 10) break;
                        if (trials >= sample_budget && trials % 100 == 0) {
                            double rhohat = (success * 1.0 / trials);
                            long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05 / 2);
                            long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05 / 2);
                            if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25) break;
                        }
                        continue;
                    }

                    bool all_predicates_pass = true;

                    // 2) 合取短路检查：按顺序逐个查询点检查
                    for (int q_node : target_query_nodes) {
                        int mapped_data_node = sample[q_node];

                        bool pass = false;
                        auto cache_it = oracle_result_cache.find(mapped_data_node);

                        if (cache_it != oracle_result_cache.end()) {
                            // 旧点：不消耗预算，且不重复调用Oracle
                            pass = cache_it->second;
                            oracle_cache_hits++;
                        } else {
                            // 新点：先做预算检查，再调用Oracle
                            if (use_custom_budget) {
                                if (budget_unique_nodes.size() >= (size_t)custom_oracle_budget) {
                                    goto END_ESTIMATION;
                                }
                            }

                            budget_unique_nodes.insert(mapped_data_node);
                            oracle_budget_used++;

                            pass = oracle_manager.CheckOracle(mapped_data_node, 0.5);
                            oracle_calls++;
                            oracle_result_cache[mapped_data_node] = pass;
                        }

                        if (!pass) {
                            all_predicates_pass = false;
                            break; // 合取短路：该样本后续点不再检查
                        }
                    }

                    trials++;
                    if (all_predicates_pass) {
                        success++;
                        double g = 1.0;
                        if (agg_func == AGG_SUM) {
                            g = 0.0;
                            for (int node : agg_query_nodes) {
                                g += oracle_manager.GetAggValue(sample[node]);
                            }
                        }
                        agg_success_sum += g;
                    }

                    double rhohat = (success * 1.0 / trials);
                    if (trials == 50000 && success <= 10) break;
                    if (trials >= sample_budget && trials % 100 == 0) {
                        long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, 0.05 / 2);
                        long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, 0.05 / 2);
                        if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25) break;
                    }
                }

            END_ESTIMATION:
                double est = (trials > 0) ? (agg_success_sum / trials) * total_trees_ : 0.0;

                timer.Stop();
                info["QueryTime"] = timer.GetTime();
                info["#TreeTrials"] = trials;
                info["#TreeSuccess"] = success;
                info["#UniqueOracleNodes"] = (double)budget_unique_nodes.size();

                // 新增调试统计
                info["#OracleCalls"] = (double)oracle_calls;
                info["#OracleCacheHits"] = (double)oracle_cache_hits;
                info["#OracleBudgetUsed"] = (double)oracle_budget_used;

                return est;
            }

            std::unordered_map<int, double> EstimateContainingNode(const std::vector<int> &sv) {
                Timer timer; timer.Start();
                std::unordered_map<int, double> result;

                if (!query_ || !CS) {
                    std::cerr << "[Error] CandidateTreeSampler not initialized.\n";
                    return result;
                }

                if (sv.empty()) return result;

                // 1) 确保所有 sv id 合法
                for (int node_id : sv) {
                    if (node_id < 0 || node_id >= data_->GetNumVertices()) {
                        std::cerr << "[Warning] Invalid node id in sv: " << node_id
                                << " (graph has " << data_->GetNumVertices() << " vertices). Skipping.\n";
                    }
                }

                // 2) 查出 sv 的标签，假设它们同标签（如你所述）
                int labelL = data_->GetVertexLabel(sv[0]);

                // 3) 找出 query 中第一个 label == labelL 的节点（若 query 中有多个相同标签，你也可以扩展为处理所有）
                int query_node_L = -1;
                for (int qv = 0; qv < query_->GetNumVertices(); qv++) {
                    if (query_->GetVertexLabel(qv) == labelL) {
                        query_node_L = qv;
                        break;
                    }
                }
                if (query_node_L == -1) {
                    std::cerr << "[Warning] No query node has label " << labelL << "!\n";
                    for (int node_id : sv) result[node_id] = 0.0;
                    return result;
                }

                // 4) 如果当前 Tq.root 不是 query_node_L, 需要重建 Tq 并重新计算 CountCandidateTrees()
                if (Tq.root != query_node_L) {
                    // Rebuild spanning tree (use existing edges of query)
                    Tq.Initialize(query_, query_node_L);
                    for (int u = 0; u < query_->GetNumVertices(); u++) {
                        for (int v : query_->GetNeighbors(u)) {
                            if (u < v) Tq.AddEdge(u, v);
                        }
                    }
                    Tq.BuildTree();

                    // 关键：重新计算依赖于树结构的内部数据（num_trees_, sample_dist_, ...）
                    CountCandidateTrees();
                }

                // 5) 对每个 sv_i 做自适应采样（与原 Estimate() 的逻辑一致）
                for (int node_id : sv) {
                    if (node_id < 0 || node_id >= data_->GetNumVertices()) {
                        result[node_id] = 0.0;
                        continue;
                    }

                    // 在候选集中找到对应的 cand_idx（query_node_L 下）
                    int cand_idx = -1;
                    int cand_size = CS->GetCandidateSetSize(query_node_L);
                    for (int i = 0; i < cand_size; ++i) {
                        if (CS->GetCandidate(query_node_L, i) == node_id) {
                            cand_idx = i;
                            break;
                        }
                    }
                    if (cand_idx == -1) { result[node_id] = 0.0; continue; }

                    // 自适应采样
                    int success = 0, trials = 0;
                    const int MAX_TRIALS = 50000;
                    const double ALPHA = 0.05;

                    // 进行采样循环；每次都基于 sample_dist_（已经由 CountCandidateTrees 构造）
                    while (++trials) {
                        // 初始化 sample & seen
                        std::fill(sample.begin(), sample.end(), -1);
                        std::memset(seen, 0, sizeof(bool) * data_->GetNumVertices());

                        // 固定根 candidate
                        sample[Tq.root] = cand_idx;
                        seen[node_id] = true;

                        bool valid = true;
                        for (int i = 0; i < query_->GetNumVertices(); ++i) {
                            int u = Tq.GetKthVertex(i);
                            int v_idx = sample[u];
                            if (v_idx < 0) continue;
                            auto &children = Tq.GetChildren(u);
                            for (int uc_idx = 0; uc_idx < (int)children.size(); ++uc_idx) {
                                int uc = children[uc_idx];

                                // 注意：确保 sample_dist_ 的结构已经充分被初始化（由 CountCandidateTrees 保证）
                                if (sample_dist_.size() <= (size_t)u ||
                                    sample_dist_[u].size() <= (size_t)v_idx ||
                                    sample_dist_[u][v_idx].size() <= (size_t)uc_idx) {
                                    // 出现不匹配（应当不会），保守地认为无效采样
                                    valid = false;
                                    goto CLEANUP_INJECTIVE;
                                }

                                int vc_idx = sample_from_distribution(sample_dist_[u][v_idx][uc_idx]);
                                sample[uc] = CS->GetCandidateNeighbor(u, v_idx, uc, vc_idx);
                                int cand = CS->GetCandidate(uc, sample[uc]);
                                if (seen[cand]) { valid = false; goto CLEANUP_INJECTIVE; }
                                seen[cand] = true;
                            }
                        }

                        CLEANUP_INJECTIVE:
                        // clean seen
                        for (int i = 0; i < query_->GetNumVertices(); ++i) {
                            if (sample[i] >= 0) {
                                int cand = CS->GetCandidate(i, sample[i]);
                                if (cand >= 0 && cand < data_->GetNumVertices()) seen[cand] = false;
                            }
                        }

                        if (valid) success++;

                        // adaptive stop checks
                        double rhohat = (success * 1.0 / trials);
                        if (trials == MAX_TRIALS && success <= 10) {
                            result[node_id] = -1.0; // indicate unreliable
                            break;
                        }
                        if (trials >= 1000 && trials % 100 == 0) {
                            long double wplus = boost::math::binomial_distribution<>::find_upper_bound_on_p(trials, success, ALPHA/2);
                            long double wminus = boost::math::binomial_distribution<>::find_lower_bound_on_p(trials, success, ALPHA/2);
                            if (rhohat * 0.8 < wminus && wplus < rhohat * 1.25) {
                                break;
                            }
                        }
                    } // end trials loop

                    double final_rho = (trials > 0) ? (success * 1.0 / trials) : 0.0;
                    double est_partial = final_rho * num_trees_[Tq.root][cand_idx];
                    result[node_id] = est_partial;
                } // end for each sv

                timer.Stop();
                info["TreeSampleTime"] = timer.GetTime();
                return result;
            }
        };
    }
}