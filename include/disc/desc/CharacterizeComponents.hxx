#pragma once

#include <disc/desc/Desc.hxx>
#include <disc/desc/Desc.hxx>
#include <disc/desc/Encoding.hxx>

namespace sd
{
namespace disc
{

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_one_component(Composition<Trait>& c,
                                size_t              index,
                                const Config&       cfg,
                                Interface&&         f = {})
{
    c.models[index] = make_distribution(c, cfg);
    c.assignment[index].clear();

    for (size_t i = 0; i < c.summary.size(); ++i)
    {
        const auto& x = c.summary.point(i);
        if (is_singleton(x))
        {
            c.assignment[index].insert(i);
            c.models[index].insert_singleton(c.frequency(i, index), x, false);
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
        if (f.confidence(c, index, q, x, cfg)) // assignment_score
        {
            c.assignment[index].insert(i);
            c.models[index].insert(q, x, true);
        }
    }
}

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_no_mining(Composition<Trait>& c, const Config& cfg, Interface&& f = {})
{
    compute_frequency_matrix(c);

    c.assignment.assign(c.data.num_components(), {});
    c.models.assign(c.assignment.size(), make_distribution(c, cfg));
    // c.subset_encodings.assign(c.data.num_components(), {});

    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        characterize_one_component(c, j, cfg, f);
    }
}

// template <typename Trait, typename Interface = DefaultAssignment>
// void characterize_components(Component<Trait>& c, const Config& cfg, Interface&& f = {})
// {
//     characterize_no_mining(c, cfg, std::forward<Interface>(f));
//     c.encoding = encode(c, cfg);
// }

template <typename Trait, typename Interface = DefaultAssignment>
void characterize_components(Composition<Trait>& c, const Config& cfg, Interface&& f = {})
{
    characterize_no_mining(c, cfg, std::forward<Interface>(f));
    // c.encoding = encode(c, cfg);
}

template <typename Trait, typename Interface = DefaultAssignment>
void initialize_model(Composition<Trait>& c, const Config& cfg, Interface&& f = {})
{
    c.data.group_by_label();
    insert_missing_singletons(c.data, c.summary);
    characterize_no_mining(c, cfg, std::forward<Interface>(f));
    assert(check_invariant(c));
}


template <typename Trait, typename Interface = DefaultAssignment>
void characterize_components(Component<Trait>& c, const Config& cfg = {}, Interface&& f = {})
{
    auto& data    = c.data;
    auto& summary = c.summary;
    assert(c.model.model.dim == data.dim);
    assert(c.model.model.factors.size() == data.dim);

    auto& pr = c.model;

    for (const auto& i : summary)
    {
        if (is_singleton(point(i)))
        {
            pr.insert_singleton(label(i), point(i), false);
        }
    }

    for (const auto& i : summary)
    {
        const auto& x = point(i);
        const auto& q = label(i);

        if (q == 0 || is_singleton(x) || !c.model.is_allowed(x))
            continue;

        if (f.confidence(c, q, x, cfg)) // assignment_score
        {
            c.model.insert(label(i), point(i), true);
        }
    }
    
    // estimate_model(pr);

    return pr;
}

template <typename Trait, typename Interface = DefaultAssignment>
void initialize_model(Component<Trait>& c, const Config& cfg, Interface&& f)
{
    auto& data    = c.data;
    auto& summary = c.summary;
    insert_missing_singletons(data, summary);
    data.dim = std::max(data.dim, summary.dim);
    c.model = make_distribution(c, cfg);
    characterize_components(c, cfg, std::forward<Interface>(f));
}

} // namespace disc
} // namespace sd
