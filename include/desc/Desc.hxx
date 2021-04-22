#pragma once

#include <desc/CharacterizeComponents.hxx>
#include <desc/Component.hxx>
#include <desc/Composition.hxx>
#include <desc/PatternAssignment.hxx>
#include <desc/PatternsetMiner.hxx>
#include <desc/utilities/ModelPruning.hxx>

namespace sd::disc
{

struct IDesc : DefaultPatternsetMinerInterface, DefaultAssignment
{
    template <typename C, typename Config>
    static auto finish(C& c, const Config& cfg)
    {
        prune_composition(c, cfg, DefaultAssignment{});
        // using dist_t = typename std::decay_t<C>::distribution_type;
        // if constexpr (is_dynamic_factor_model<dist_t>())
        // {
        //     relent_prune_pattern_composition(c, cfg);
        // }
        // else
        // {
        //     prune_pattern_composition(c, cfg);
        // }
    }
};
} // namespace sd::disc
