#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/PatternsetMinerGeneric.hxx>
#include <disc/disc/PatternsetResult.hxx>

namespace sd::disc
{

template <typename Trait, typename CALLBACK = EmptyCallback>
Composition<Trait>
discover_patternsets(Composition<Trait> c, const MiningSettings& cfg, CALLBACK&& callback = {})
{
    discover_patterns_generic(c, cfg, {}, std::forward<CALLBACK>(callback));
    return c;
}

template <typename Trait, typename CALLBACK = EmptyCallback>
PatternsetResult<Trait> discover_patternset(PatternsetResult<Trait> c,
                                            const MiningSettings&   cfg,
                                            CALLBACK&&              callback = {})
{
    discover_patterns_generic(c, cfg, {}, std::forward<CALLBACK>(callback));
    return c;
}

} // namespace sd::disc