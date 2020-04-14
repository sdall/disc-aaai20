#pragma once

#include <disc/disc/Desc.hxx>
#include <disc/disc/Encoding.hxx>

namespace sd
{
namespace disc
{

template <typename Trait, typename Pattern>
bool reassign_candidate_from_summary(Composition<Trait>& c,
                                     size_t              candidate_index,
                                     const Pattern&      candidate_pattern,
                                     bool                bic)
{
    size_t assignment_count = 0;

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        const auto q = c.frequency(candidate_index, i);
#if 1
        const bool is_extremely_frequent = q > 0.7;
        if (bic && is_extremely_frequent && c.models[i].is_itemset_allowed(candidate_pattern))
        {
            c.assignment[i].insert(candidate_index);
            c.models[i].insert(q, candidate_pattern, true);
            assignment_count += 1;
        }
        else
        {
            assignment_count +=
                find_assignment_if(c, candidate_index, candidate_pattern, i, q, bic, true);
        }
#else
        assignment_count +=
            find_assignment_if(c, candidate_index, candidate_pattern, i, q, bic, true);
#endif
    }
    return assignment_count > 0;
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
void characterize_no_mining(Composition<Trait>& c, const MiningSettings& cfg)
{
    if (c.frequency.length() != c.summary.size() ||
        c.frequency.extent(1) < c.data.num_components())
    {
        compute_frequency_matrix(c.data, c.summary, c.frequency);
    }

    c.assignment.assign(c.data.num_components(), {});
    c.models.assign(c.assignment.size(), make_distribution(c, cfg));
    c.subset_encodings.resize(c.data.num_components());

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
            reassign_candidate_from_summary(c, candidate_index, x, cfg.use_bic);
        }
    }

    // retrain_models(c);
    c.initial_encoding = c.encoding;
    c.encoding         = encode(c, cfg, false);
}

template <typename Trait>
void initialize_composition(Composition<Trait>& c, const MiningSettings& cfg)
{
    c.data.group_by_label();

    if (cfg.with_singletons)
    {
        insert_missing_singletons(c.data, c.summary);
    }

    characterize_no_mining(c, cfg);

    assert(check_invariant(c));
}

} // namespace disc
} // namespace sd