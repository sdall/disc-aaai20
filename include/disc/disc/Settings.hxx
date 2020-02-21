#pragma once

#include <chrono>
#include <nonstd/optional.hpp>

namespace sd
{
namespace disc
{

struct MiningSettings
{
    // bool   quick           = true;
    double alpha           = 0.01;
    bool   with_singletons = true;
    bool   use_bic         = true;
    size_t max_iteration   = std::numeric_limits<size_t>::max();
    size_t min_support     = 1;
    size_t max_patience    = 5;

    nonstd::optional<size_t>                    max_pattern_size;
    nonstd::optional<size_t>                    max_patternset_size;
    nonstd::optional<std::chrono::milliseconds> max_time;

    struct
    {
        size_t max_width_per_factor    = 5; // 5
        size_t max_num_sets_per_factor = 8; // 8
    } distribution;

    struct RelaxedDistribution
    {
        size_t budget_limit = 8;
        size_t mode         = 5;
        // viva::RelaxationMode kind = viva::RelaxationMode::weak;
        // viva::RelaxationMode kind = viva::RelaxationMode::weak;
    };
    nonstd::optional<RelaxedDistribution> relaxed_distribution;
};

struct DecompositionSettings : public MiningSettings
{
    nonstd::optional<size_t>               max_cliques;
    nonstd::optional<size_t>               max_em_iterations; // per trial
    nonstd::optional<std::chrono::seconds> max_time_total;

    bool test_divergence     = true;
    bool intermediate_mining = false;
    // `intermediate_mining` option is set to true in our method: ``mine_split_round_repeat''
};

} // namespace disc
} // namespace sd