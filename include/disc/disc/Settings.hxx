#pragma once

#include <disc/desc/Settings.hxx>

namespace sd::disc
{

struct DiscConfig : public Config
{
    std::optional<size_t>                    max_cliques;
    std::optional<size_t>                    max_em_iterations;
    std::optional<std::chrono::milliseconds> max_time_total;
    bool                                     test_divergence = true;
};

} // namespace sd::disc