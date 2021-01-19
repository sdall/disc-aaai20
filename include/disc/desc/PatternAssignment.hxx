#pragma once

#include <disc/desc/Composition.hxx>
#include <disc/desc/Encoding.hxx>
#include <disc/desc/Settings.hxx>

namespace sd
{
namespace disc
{

struct DefaultAssignment
{
    template <typename Trait, typename Pattern>
    static auto confidence(const Composition<Trait>&         c,
                           size_t                            index,
                           const typename Trait::float_type& q,
                           const Pattern&                    x,
                           const Config&)
    {
        using std::log2;
        return log2(q / c.models[index].expectation(x)) > 0;
    }
    
    template <typename Trait, typename Pattern>
    static auto confidence(const Component<Trait>&         c,
                           const typename Trait::float_type& q,
                           const Pattern&                    x,
                           const Config&)
    {
        using std::log2;
        return log2(q / c.model.expectation(x)) > 0;
    }
};

template <typename Trait, typename Candidate>
void insert_pattern_to_summary(Composition<Trait>& c, const Candidate& x)
{
    using float_type = typename Trait::float_type;

    assert(c.frequency.extent(1) > 0);

    const auto glob_frequency = static_cast<float_type>(x.support) / c.data.size();
    c.summary.insert(glob_frequency, x.pattern);

    c.frequency.push_back();
    auto new_q = c.frequency[c.frequency.extent(0) - 1];
    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        auto s   = size_of_intersection(x.row_ids, c.masks[j]);
        new_q(j) = static_cast<float_type>(s) / c.data.subset(j).size();
    }
    // new_q.back() = glob_frequency;
}

template <typename Trait, typename Candidate, typename Interface = DefaultAssignment>
bool find_assignment_impl(Composition<Trait>& c,
                          const Candidate&    x,
                          const Config&       cfg,
                          Interface&&         f = {})
{
    using float_type = typename Trait::float_type;

    if (x.score < 0)
        return false;

    size_t counter = 0;
    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        auto  s  = size_of_intersection(x.row_ids, c.masks[i]);
        auto& pr = c.models[i];
        if (s > 0 && pr.is_allowed(x.pattern))
        {
            auto q = static_cast<float_type>(s) / c.data.subset(i).size();

            if (f.confidence(c, i, q, x.pattern, cfg))
            {
                pr.insert(q, x.pattern, true);
                c.assignment[i].insert(c.summary.size());
                ++counter;
            }
        }
    }

    if (counter)
    {
        insert_pattern_to_summary(c, x);
        return true;
    }
    else
    {
        return false;
    }
}

template <typename Trait, typename Candidate>
bool find_assignment_impl_first(Composition<Trait>& c, const Candidate& x, const Config&)
{
    using float_type = typename Trait::float_type;

    if (x.score <= 0 || x.support == 0 || !c.models[0].is_allowed(x.pattern))
        return false;

    auto q = static_cast<float_type>(x.support) / c.data.size();

    c.models[0].insert(q, x.pattern, true);
    c.assignment[0].insert(c.summary.size());
    insert_pattern_to_summary(c, x);

    return true;
}

template <typename Trait, typename Candidate, typename Interface = DefaultAssignment>
bool find_assignment(Composition<Trait>& c,
                     const Candidate&    x,
                     const Config&       cfg,
                     Interface&&         f = {})
{
    if (c.data.num_components() == 1)
    {
        return find_assignment_impl_first(c, x, cfg);
    }
    else
    {
        return find_assignment_impl(c, x, cfg, std::forward<Interface>(f));
    }
}

template <typename Trait, typename Candidate, typename Interface = DefaultAssignment>
bool find_assignment(Component<Trait>& c, const Candidate& x, const Config&, Interface&& = {})
{
    using float_type = typename Trait::float_type;

    if (x.score < 0)
        return false;

    auto q = static_cast<float_type>(x.support) / c.data.size();

    c.model.insert(q, x.pattern, true);
    c.summary.insert(q, x.pattern);

    return true;
}

} // namespace disc
} // namespace sd