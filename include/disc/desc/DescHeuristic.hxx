#pragma once

#include <disc/desc/Composition.hxx>
#include <disc/desc/Encoding.hxx>

namespace sd::disc
{

template <typename Trait, typename Candidate>
auto onetime_cost_pattern(const Composition<Trait>& c, const Candidate& x, const Config& cfg) ->
    typename Trait::float_type
{
    if (cfg.use_bic)
    {
        return additional_cost_bic(x.pattern, c.data.size(), c.models.front());
    }
    else
    {
        return constant_mdl_cost(c, x.pattern);
    }
}

template <typename Trait, typename Candidate>
auto additional_cost_pattern(const Composition<Trait>& c,
                             const Candidate&,
                             size_t        support,
                             size_t        component,
                             const Config& cfg) -> typename Trait::float_type
{
    if (cfg.use_bic)
    {
        return 0;
    }
    else
    {
        return additional_cost_mdl(support, c.models[component]);
    }
}

template <typename C, typename Dist, typename Candidate>
auto total_cost_pattern(const C& c, const Dist& dist, const Candidate& x, const Config& cfg) ->
    typename C::float_type
{
    if (cfg.use_bic)
    {
        return additional_cost_bic(x.pattern, c.data.size(), dist);
    }
    else
    {
        return constant_mdl_cost(c, x.pattern) + additional_cost_mdl(x.support, dist);
    }
}

template <typename Trait, typename Candidate, typename Masks>
auto desc_heuristic_multi(const Composition<Trait>& c,
                          const Candidate&          x,
                          const Masks&              masks,
                          const Config&             cfg)
{
    using float_type = typename Trait::float_type;

    float_type acc = -onetime_cost_pattern(c, x, cfg);

    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        auto p = c.models[i].expectation(x.pattern);
        auto s = size_of_intersection(x.row_ids, masks[i]);
        auto q = static_cast<float_type>(s) / c.data.subset(i).size();
        auto h = s == 0 ? 0 : s * std::log2(q / p);

        assert(0 <= p && p <= 1);

        acc += h - additional_cost_pattern(c, x, s, i, cfg);
    }

    return acc;
}

template <typename C, typename Distribution, typename Candidate>
auto desc_heuristic_1(const C& c, const Distribution& pr, const Candidate& x, const Config& cfg)
{
    using float_type = typename C::float_type;

    const auto s = static_cast<float_type>(x.support);
    const auto q = s / c.data.size();
    const auto p = pr.expectation(x.pattern);

    assert(0 <= p && p <= 1);

    const float_type h = s == 0 ? 0 : s * std::log2(q / p);

    return h - total_cost_pattern(c, pr, x, cfg);
}

template <typename Trait, typename Candidate>
auto desc_heuristic(const Composition<Trait>& c, const Candidate& x, const Config& cfg) ->
    typename Trait::float_type
{
    if (c.data.num_components() == 1)
    {
        return desc_heuristic_1(c, c.models.front(), x, cfg);
    }
    else
    {
        return desc_heuristic_multi(c, x, c.masks, cfg);
    }
}

template <typename Trait, typename Candidate>
auto desc_heuristic(const Component<Trait>& c, const Candidate& x, const Config& cfg) ->
    typename Trait::float_type
{
    return desc_heuristic_1(c, c.model, x, cfg);
}

} // namespace sd::disc