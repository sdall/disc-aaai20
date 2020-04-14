#pragma once

#include <chrono>
#include <optional>

namespace sd
{
namespace disc
{

struct MiningSettings
{
    bool   with_singletons = true;
    bool   use_bic         = true;
    size_t min_support     = 2;
    size_t max_patience    = 9;
    size_t max_iteration   = std::numeric_limits<size_t>::max();

    std::optional<size_t>                    max_pattern_size;
    std::optional<size_t>                    max_patternset_size;
    std::optional<std::chrono::milliseconds> max_time;

    struct
    {
        size_t max_width_per_factor    = 20; // 5
        size_t max_num_sets_per_factor = 8;  // 8
    } distribution;
};

struct DecompositionSettings : public MiningSettings
{
    std::optional<size_t>                    max_cliques;
    std::optional<size_t>                    max_em_iterations;
    std::optional<std::chrono::milliseconds> max_time_total;
    bool                                     test_divergence = true;
    double                                   alpha           = 0.01;
};

} // namespace disc
} // namespace sd