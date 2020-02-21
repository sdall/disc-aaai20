#pragma once

#include <disc/disc/DiscoverPatternset.hxx>
#include <disc/disc/DiscoverPatternsets.hxx>
#include <disc/disc/Encoding.hxx>

namespace sd
{
namespace disc
{

template <typename S>
auto make_distribution(S const& c, MiningSettings const& cfg)
{
    using distribution_type = typename S::distribution_type;
    // using float_type        = typename S::float_type;
    // using pattern_type      = typename S::pattern_type;

    return distribution_type(c.data.dim, c.data.size(), cfg);

    // auto r = distribution_type(c.data.dim,
    //                            c.data.size(),
    //                            cfg.distribution.max_width_per_factor,
    //                            cfg.distribution.max_num_sets_per_factor);

    // if constexpr (!std::is_same_v<distribution_type, MEDistribution<pattern_type,
    // float_type>>)
    // {
    //     r.budget_limit = cfg.relaxed_distribution->budget_limit;
    //     r.mode = static_cast<std::decay_t<decltype(r.mode)>>(cfg.relaxed_distribution->mode);
    // }

    // return r;
}

template <typename Trait>
void retrain_models(Composition<Trait>& c)
{
    for (auto& m : c.models)
    {
        estimate_model(m);
    }
}

template <typename Trait>
void recreate_models(Composition<Trait>& c, MiningSettings const& cfg)
{
    const size_t n = c.data.num_components();

    c.models.assign(n, make_distribution(c, cfg));

#pragma omp parallel for
    for (size_t j = 0; j < n; ++j)
    {
        for (auto i : c.assignment[j])
        {
            c.models[j].insert(c.frequency(i, j), c.summary.point(i));
        }
        estimate_model(c.models[j]);
    }
}

template <typename Trait>
Composition<Trait> characterize_no_mining_actual_gain(Composition<Trait>    c,
                                                      const MiningSettings& cfg)
{
    c.assignment = AssignmentMatrix(c.data.num_components());
    if (c.frequency.length() != c.summary.size() &&
        c.frequency.extent(1) < c.data.num_components())
    {
        compute_frequency_matrix(c.data, c.summary, c.frequency);
    }

    for (size_t candidate_index = 0; candidate_index < c.summary.size(); ++candidate_index)
    {
        const auto& x = c.summary.point(candidate_index);
        if (is_singleton(x))
        {
            for (auto& a : c.assignment)
            {
                a.insert(candidate_index);
            }
        }
    }

    recreate_models(c, cfg);

    c.subset_encodings = make_data_encodings(c, c.models);

    // separate stages: depends on singletons.
    for (size_t candidate_index = 0; candidate_index < c.summary.size(); ++candidate_index)
    {
        const auto& x = c.summary.point(candidate_index);
        if (!is_singleton(x))
        {
            c = find_assignment_actual_gain(std::move(c), candidate_index, x, cfg.use_bic)
                    .second;
        }
    }

    return c;
}

template <typename Trait, typename Pattern, typename float_type>
bool reassign_candidate_from_summary(Composition<Trait>& c,
                                     size_t              candidate_index,
                                     const Pattern&      candidate_pattern,
                                     bool                bic,
                                     float_type          alpha)
{
    size_t assignment_count = 0;

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        const auto q = c.frequency(candidate_index, i);
#if 1
        const bool is_extremely_frequent = q > 0.7;
        if (bic && is_extremely_frequent && c.models[i].is_item_allowed(candidate_pattern))
        {
            c.assignment[i].insert(candidate_index);
            c.models[i].insert(q, candidate_pattern, true);
            assignment_count += 1;
        }
        else
        {
            assignment_count += find_assignment_if(
                c, candidate_index, candidate_pattern, i, q, bic, float_type(alpha), true);
        }
#else
        assignment_count += find_assignment_if(
            c, candidate_index, candidate_pattern, i, q, bic, float_type(alpha), true);
#endif
    }
    return assignment_count > 0;
}

template <typename Trait>
void characterize_no_mining(Composition<Trait>& c, const MiningSettings& cfg)
{
    if (c.frequency.length() != c.summary.size() ||
        c.frequency.extent(1) < c.data.num_components())
    {
        compute_frequency_matrix(c.data, c.summary, c.frequency);
    }

    c.assignment.assign(c.data.num_components(), {});
    c.models.assign(c.assignment.size(), make_distribution(c, cfg));

    for (size_t candidate_index = 0; candidate_index < c.summary.size(); ++candidate_index)
    {
        const auto& x = c.summary.point(candidate_index);
        if (is_singleton(x))
        {
            for (auto& a : c.assignment)
            {
                a.insert(candidate_index);
            }
            for (size_t m = 0; m < c.models.size(); ++m)
            {
                c.models[m].insert(c.frequency(candidate_index, m), x);
            }
        }
    }

    retrain_models(c);

    // separate stages: depends on singletons.
    for (size_t candidate_index = 0; candidate_index < c.summary.size(); ++candidate_index)
    {
        const auto& x = c.summary.point(candidate_index);
        if (!is_singleton(x))
        {
            reassign_candidate_from_summary(c, candidate_index, x, cfg.use_bic, cfg.alpha);
        }
    }

    // retrain_models(c);
    c.initial_encoding = c.encoding;
    c.encoding         = encoding_length_mdm(c, cfg.use_bic);
}

template <typename Trait>
void resize_model_after_split(Composition<Trait>&       c,
                              std::pair<size_t, size_t> index,
                              const MiningSettings&     cfg)
{
    using float_type = typename Trait::float_type;
    // clang-format off
    c.assignment      .resize(c.data.num_components());
    c.models          .resize(c.data.num_components(), make_distribution(c, cfg));
    c.subset_encodings.resize(c.data.num_components(), 0);
    // clang-format on

    if (c.frequency.size() == 0 || c.data.num_components() == 1 ||
        c.frequency.extent(1) > c.data.num_components())
    {
        compute_frequency_matrix(c.data, c.summary, c.frequency);
    }
    else
    {
        ndarray<float_type, 2> q({}, c.summary.size(), c.data.num_components() + 1);
        for (auto i : c.frequency.subscripts())
        {
            q(i) = c.frequency(i);
        }
        for (size_t i = 0; i < c.frequency.extent(0); ++i)
        {
            q(i, q.extent(1) - 1) = c.frequency(i, c.frequency.extent(1) - 1);
        }
        c.frequency = std::move(q);

        assert(c.frequency.extent(1) >= c.data.num_components());
        assert(c.models.size() > index.second);
        assert(c.assignment.size() > index.second);
        assert(c.frequency.extent(1) > index.second);

        compute_frequency_matrix(c.data, c.summary, c.frequency, index.first);
        compute_frequency_matrix(c.data, c.summary, c.frequency, index.second);
    }

    assert(check_invariant(c));
}

template <typename Trait>
void characterize_one_component_no_mining(Composition<Trait>&   c,
                                          size_t                index,
                                          const MiningSettings& cfg)
{
    c.models[index] = make_distribution(c, cfg);
    c.assignment[index].clear();

    estimate_model(c.models[index]);

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        if (is_singleton(x))
        {
            c.assignment[index].insert(i);
            c.models[index].insert_singleton(c.frequency(i, index), x);
        }
    }

    estimate_model(c.models[index]);

    // separate stages: depends on singletons.
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        if (!is_singleton(x))
        {
            auto q = c.frequency(i, index);
            if (q > 0.7 && c.models[index].is_item_allowed(x))
            {
                c.assignment[index].insert(i);
                c.models[index].insert(q, x);
            }
            else
            {
                find_assignment_if(c, i, x, index, q, cfg.use_bic, cfg.alpha);
            }
        }
    }

    estimate_model(c.models[index]);
}

template <typename Trait>
Composition<Trait> characterize_split(Composition<Trait>        c,
                                      std::pair<size_t, size_t> index,
                                      const MiningSettings&     cfg)
{
    resize_model_after_split(c, index, cfg);

    retrain_models(c);

    characterize_one_component_no_mining(c, index.first, cfg);
    characterize_one_component_no_mining(c, index.second, cfg);

    c.initial_encoding = c.encoding;
    c.encoding         = encoding_length_mdm_update(c, index, cfg.use_bic);

    return c;
}

} // namespace disc
} // namespace sd