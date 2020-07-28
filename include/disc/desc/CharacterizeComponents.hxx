#pragma once

#include <disc/desc/Desc.hxx>
#include <disc/desc/Encoding.hxx>

namespace sd
{
namespace disc
{

template <typename Trait>
void characterize_one_component(Composition<Trait>& c, size_t index, const Config& cfg)
{
    c.models[index] = make_distribution(c, cfg);
    c.assignment[index].clear();

    estimate_model(c.models[index]);

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        if (is_singleton(x))
        {
            auto q = c.frequency(i, index);
            c.assignment[index].insert(i);
            c.models[index].insert_singleton(q, x, false);
        }
    }

    estimate_model(c.models[index]);

    // separate stages: depends on singletons.
    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        const auto& q = c.frequency(i, index);

        if (q == 0 || is_singleton(x) || !c.models[index].is_allowed(x))
            continue;
        if (q > 0.7 || confidence(c, index, q, x, cfg)) // assignment_score
        {
            c.assignment[index].insert(i);
            c.models[index].insert(q, x, true);
        }
    }
}

template <typename Trait>
void characterize_no_mining(Composition<Trait>& c, const Config& cfg)
{
    compute_frequency_matrix(c.data, c.summary, c.frequency);

    c.assignment.assign(c.data.num_components(), {});
    c.models.assign(c.assignment.size(), make_distribution(c, cfg));
    c.subset_encodings.resize(c.data.num_components());

    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        characterize_one_component(c, j, cfg);
    }
}

template <typename Trait>
void characterize_components(Composition<Trait>& c, const Config& cfg)
{
    characterize_no_mining(c, cfg);
    c.encoding = encode(c, cfg);
}

template <typename Trait>
void initialize_model(Composition<Trait>& c, const Config& cfg)
{
    c.data.group_by_label();
    insert_missing_singletons(c.data, c.summary);
    characterize_no_mining(c, cfg);
    assert(check_invariant(c));
}

} // namespace disc
} // namespace sd
