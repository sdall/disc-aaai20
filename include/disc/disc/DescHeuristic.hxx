#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/Encoding.hxx>

namespace sd::disc
{

template <typename Trait, typename Candidate, typename Masks>
auto mean_score_many_components(const Composition<Trait>& c,
                                const Candidate           x,
                                bool                      use_bic,
                                const Masks&              masks)
{
    using float_type = typename Trait::float_type;

    float_type acc = 0;

    if (!use_bic)
    {
        acc -= constant_mdl_cost(c, x.pattern);
    }

    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        const auto& d             = c.data.subset(i);
        auto        p             = c.models[i].expected_frequency(x.pattern);
        auto        support       = size_of_intersection(x.row_ids, masks[i]);
        auto        fr            = static_cast<float_type>(support) / d.size();
        auto        expected_gain = support * std::log2(fr / p);

        acc += expected_gain;

        if (use_bic)
        {
            acc -= additional_cost_bic(x.pattern, d.size(), c.models[i]);
        }
        else
        {
            acc -= additional_cost_mdl(support, c.models[i]);
        }
    }

    return acc;
}

template <typename C, typename Distribution, typename Candidate>
auto heuristic_score(const C& c, const Distribution& pr, Candidate& x, bool use_bic)
{
    using float_type = typename C::float_type;
    const auto fr    = static_cast<float_type>(x.support) / c.data.size();
    const auto p     = pr.expected_frequency(x.pattern);
    float_type gain  = x.support == 0 ? 0 : x.support * std::log2(fr / p);

    if (use_bic)
    {
        gain -= additional_cost_bic(x.pattern, c.data.size(), pr);
    }
    else
    {
        gain -= additional_cost_mdl(x.support, pr) + constant_mdl_cost(c, x.pattern);
    }

    return gain;
}

template <typename Trait, typename Candidate, typename Masks>
auto mean_score(const Composition<Trait>& c,
                const Candidate&          x,
                bool                      use_bic,
                const Masks&              masks) -> typename Trait::float_type
{
    if (c.data.num_components() == 1)
    {
        return heuristic_score(c, c.models.front(), x, use_bic);
    }
    else
    {
        return mean_score_many_components(c, x, use_bic, masks);
    }
}

} // namespace sd::disc