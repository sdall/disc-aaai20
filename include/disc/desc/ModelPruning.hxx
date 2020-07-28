#pragma once

#include <disc/distribution/InferProbabilities.hxx>
#include <disc/distribution/StaticFactorModel.hxx>

namespace sd::disc
{

template <typename Dist>
void prune_distribution(Dist& pr, bool estimate = true)
{
#pragma omp parallel for if (pr.factors.size() > 16)
    for (size_t i = 0; i < pr.factors.size(); ++i)
    {
        if (pr.factors[i].factor.itemsets.set.size() > 2)
        {
            viva::prune_factor(pr.factors[i], pr.max_factor_size, estimate);
        }
    }
}

template <typename Trait>
void remove_unused_patterns(disc::Component<Trait>& c, Config const& cfg)
{
    for (size_t j = 0; j < c.summary.size();)
    {
        bool keep = std::any_of(
            c.model.model.factors.begin(), c.model.model.factors.end(), [&](const auto& f) {
                return f.factor.get_precomputed_expectation(c.summary.point(j)).has_value();
            });
        if (!keep)
        {
            c.summary.erase_row(j);
        }
        else
        {
            ++j;
        }
    }
    c.encoding = disc::encode(c, cfg);
}

template <typename Trait>
void assign_from_factors(disc::Composition<Trait>& c)
{
    const auto& q = c.frequency;
    c.assignment.resize(c.data.num_components());

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        auto& a = c.assignment[i];
        auto& phi = c.models[i].model.factors;

        a.clear();

        for (size_t j = 0; j < c.summary.size(); j++)
        {
            const auto& x = c.summary.point(j);

            bool keep = is_singleton(x) ||
                        (q(j, i) > 0 && std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
                             return f.factor.get_precomputed_expectation(x).has_value();
                         }));
            if (keep)
            {
                a.insert(j);
            }
        }
    }
}

template <typename Trait>
void remove_unused_patterns_from_summary(disc::Composition<Trait>& c)
{
    for (size_t j = 0; j < c.summary.size();)
    {
        const auto& x    = c.summary.point(j);
        bool        keep = false;

        for (size_t k = 0; k < c.assignment.size(); ++k)
        {
            const auto& phi = c.models[k].model.factors;
            keep |= std::any_of(phi.begin(), phi.end(), [&](const auto& f) {
                return f.factor.get_precomputed_expectation(x).has_value();
            });

            if (keep)
                break;
        }

        if (!keep)
        {
            c.summary.erase_row(j);
        }
        else
        {
            ++j;
        }
    }
}

template <typename Trait>
void remove_unused_patterns(disc::Composition<Trait>& c, Config const& cfg)
{
    remove_unused_patterns_from_summary(c);
    // characterize_components(c, cfg);
    assign_from_factors(c);
    compute_frequency_matrix(c.data, c.summary, c.frequency);
    c.encoding = disc::encode(c, cfg);
}

template <typename Trait>
void prune_pattern_composition(disc::Component<Trait>& c,
                               Config const&           cfg)
{
    prune_distribution(c.model.model, true);
    remove_unused_patterns(c, cfg);
}

template <typename Trait>
void prune_pattern_composition(disc::Composition<Trait>& c,
                               Config const&             cfg)

{
    for (auto& m : c.models)
        prune_distribution(m.model, true);
    remove_unused_patterns(c, cfg);
}

} // namespace sd::disc