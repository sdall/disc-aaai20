#pragma once

#include <disc/disc/AssignPatternToComponents.hxx>
#include <disc/disc/DescHeuristic.hxx>
#include <disc/disc/Encoding.hxx>
#include <disc/disc/SlimCandidateGeneration.hxx>
#include <disc/utilities/EmptyCallback.hxx>

namespace sd::disc
{

template <typename Trait, typename Candidate, typename Config>
auto heuristic(PatternsetResult<Trait>& c, Candidate& x, const Config& cfg)
{
    return heuristic_score(c, c.model, x, cfg.use_bic);
}
template <typename Trait, typename Candidate, typename Config>
auto heuristic(Composition<Trait>& c, Candidate& x, const Config& cfg)
{
    return mean_score(c, x, cfg.use_bic, c.masks);
}

template <typename Trait, typename Candidate, typename Config>
bool insert_into_model(PatternsetResult<Trait>& c, Candidate& x, const Config&)
{
    using float_type = typename Trait::float_type;

    if (x.score < 0)
        return false;

    const auto fr = static_cast<float_type>(x.support) / c.data.size();
    auto&      pr = c.model;
    pr.insert(fr, x.pattern, true);
    c.summary.insert(fr, x.pattern);
    return true;
}
template <typename Trait, typename Candidate, typename Config>
bool insert_into_model(Composition<Trait>& c, Candidate& x, const Config& cfg)
{
    return find_assignment(c, x, c.masks, cfg.use_bic);
}

template <typename Trait, typename Pattern>
bool is_itemset_allowed(const PatternsetResult<Trait>& c, const Pattern& x)
{
    return c.model.is_itemset_allowed(x);
}
template <typename Trait, typename Pattern>
bool is_itemset_allowed(const Composition<Trait>& c, const Pattern& x)
{
    return std::any_of(
        begin(c.models), end(c.models), [&](const auto& m) { return m.is_itemset_allowed(x); });
}

template <typename Trait, typename Config>
void prepare(PatternsetResult<Trait>& c, const Config& cfg)
{
    if (c.summary.empty() || c.model.model.dim != c.data.dim)
        initialize_model(c, cfg);
}
template <typename Trait, typename Config>
void prepare(Composition<Trait>& c, const Config&)
{
    assert(check_invariant(c));
    c.masks = construct_component_masks(c);
}

struct DefaultPatternsetMinerInterface
{
    template <typename C, typename Config>
    static decltype(auto) objective(C& c, const Config& cfg)
    {
        return sd::disc::encode(c, cfg);
    }
    template <typename C, typename Pattern, typename Config>
    static decltype(auto) heuristic(C& c, Pattern& x, const Config& cfg)
    {
        return sd::disc::heuristic(c, x, cfg);
    }
    template <typename C, typename Config>
    static decltype(auto) prepare(C& c, const Config& cfg)
    {
        return sd::disc::prepare(c, cfg);
    }
    template <typename C, typename Config>
    static decltype(auto) finish(C&, const Config&)
    {
    }
    template <typename C, typename Pattern>
    static decltype(auto) is_itemset_allowed(C& c, const Pattern& x)
    {
        return sd::disc::is_itemset_allowed(c, x);
    }
    template <typename C, typename Candidate, typename Config>
    static decltype(auto) insert_into_model(C& c, Candidate& x, const Config& cfg)
    {
        return sd::disc::insert_into_model(c, x, cfg);
    }
};

template <typename C,
          typename I    = DefaultPatternsetMinerInterface,
          typename Info = EmptyCallback>
void discover_patterns_generic(C&                    s,
                               const MiningSettings& cfg,
                               [[maybe_unused]] I&& = {},
                               Info&& info          = {})
{
    using patter_type = typename C::pattern_type;
    using float_type  = typename C::float_type;
    using generator   = SlimGenerator<patter_type, float_type>;
    using clk         = std::chrono::high_resolution_clock;

    assert(s.data.dim != 0);

    I::prepare(s, cfg);
    s.encoding         = I::objective(s, cfg);
    s.initial_encoding = s.encoding;

    info(std::as_const(s));

    auto score_fn = [&](auto& x) { return I::heuristic(s, x, cfg); };
    auto gen      = generator(s.data, cfg.min_support, cfg.max_pattern_size, score_fn);

    size_t     items_used = 0;
    size_t     patience   = cfg.max_patience;
    const auto start_time = clk::now();

    for (size_t it = 0; it < cfg.max_iteration; ++it)
    {
        if (!gen.has_next())
            break;

        if (auto c = gen.next(); c && I::insert_into_model(s, *c, cfg))
        {
            gen.add_next(*c, score_fn);
            gen.prune([&](const auto& t) {
                return t.score <= 0 || !I::is_itemset_allowed(s, t.pattern);
            });

            patience   = std::max(patience * 2, cfg.max_patience);
            items_used = items_used + 1;

            info(std::as_const(s));
        }
        else if (patience-- == 0)
            break;

        if (cfg.max_time && clk::now() - start_time > *cfg.max_time)
        {
            break;
        }

        if (cfg.max_patternset_size && items_used > *cfg.max_patternset_size)
        {
            break;
        }
    }

    I::finish(s, cfg);
    s.encoding = I::objective(s, cfg);
}

} // namespace sd::disc