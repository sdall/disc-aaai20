#pragma once

#include <cmath>
#include <disc/disc/Encoding.hxx>
#include <disc/disc/SlimCandidateGeneration.hxx>

namespace sd::disc
{

template <typename value_type>
inline value_type kl1(value_type q, value_type p) noexcept
{
    return q < 1e-15 ? 0 : q * std::log2(q / p);
}

template <typename T>
T js1(const T& p, const T& q) noexcept
{
    auto m = (p + q) / T(2);
    return m == 0 ? 0 : (kl1(p, m) + kl1(q, m)) / T(2);
}

template <typename Trait, typename V, typename Masks>
auto mean_score_many_components(const Composition<Trait>& c,
                                const V&                  x,
                                bool                      use_bic,
                                const Masks&              masks)
{
    using float_type = typename Trait::float_type;

    float_type acc = 0;

    const auto constant_cost = use_bic ? additional_cost_bic(c) : float_type(0);

    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        const auto& d             = c.data.subset(i);
        auto        estimate      = c.models[i].expected_frequency(x.pattern);
        auto        support       = size_of_intersection(x.row_ids, masks[i]);
        auto        fr            = static_cast<float_type>(support) / d.size();
        auto        expected_gain = support * std::log2(fr / estimate);

        if (!use_bic)
        {
            expected_gain -= additional_cost_mdl(c, x.pattern, support, i, d.size());
        }
        acc += expected_gain;
    }

    return acc - constant_cost;
}

template <typename Trait, typename... Args, typename Masks>
auto mean_score(const Composition<Trait>&     c,
                const SlimCandidate<Args...>& x,
                bool                          use_bic,
                const Masks&                  masks) -> typename Trait::float_type
{
    if (c.data.num_components() == 1)
    {
        using float_type = typename Trait::float_type;

        const auto cost = use_bic
                              ? additional_cost_bic(c)
                              : additional_cost_mdl(c, x.pattern, x.support, 0, c.data.size());

        auto p = c.models[0].expected_frequency(x.pattern);
        auto q = float_type(x.support) / c.data.size();
        return x.support * std::log2(q / p) - cost;
    }
    else
    {
        return mean_score_many_components(c, x, use_bic, masks);
    }
}

} // namespace sd::disc