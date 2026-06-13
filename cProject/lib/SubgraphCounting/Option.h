#pragma once
#include "SubgraphMatching/CandidateSpace.h"
namespace GraphLib {
    namespace CardinalityEstimation {
        enum TreegenerationStrategy {
            TREEGEN_EDGE_MST, 
            TREEGEN_DENSITY_MST,
            TREEGEN_RANDOM
        };
        enum AggFunc {
            AGG_COUNT = 0,
            AGG_SUM   = 1
        };
        class CardEstOption : public SubgraphMatching::SubgraphMatchingOption {
        public:
            int ub_initial = 100000;
            double strata_ratio = 0.5;
            int treegen_strategy = TREEGEN_DENSITY_MST;
            // === 新增 (默认 -1 表示不指定) ===
            int root_label = -1;
            int root_query_index = -1;
            int sample_budget = 20000;
        };
    }
}