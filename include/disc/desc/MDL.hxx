#pragma once

#include <disc/desc/Composition.hxx>
#include <disc/desc/Support.hxx>
#include <disc/distribution/Distribution.hxx>
#include <disc/storage/Dataset.hxx>

#include <math/nchoosek.hxx>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sd::disc::mdl
{

constexpr size_t iterated_log2(double n)
{
    // iterated log
    if (n <= 1)
        return 0;
    if (n <= 2)
        return 1;
    if (n <= 4)
        return 2;
    if (n <= 16)
        return 3;
    if (n <= 65536)
        return 4;
    return 5; // if (n <= 2^65536)
    // never 6 for our case.
}

constexpr double universal_code(size_t n) { return iterated_log2(n) + 1.5186; }
// 1.5186 = log2(2.865064)

template <typename value_type, typename itemset_data>
value_type encode_summary_mtv(const itemset_data& summary, size_t data_size, size_t data_dim)
{
    const value_type log_n = std::log2(data_size);
    const value_type log_m = std::log2(data_dim);
    const value_type n1    = std::accumulate(
        summary.begin(), summary.end(), value_type{0}, [&](auto acc, const auto& i) {
            return acc + count(point(i));
        });
    return (log_n + 1 + 1.4427) * summary.size() + log_m * n1;
}

auto encode_num_component(size_t num_subsets) { return universal_code(num_subsets); }

auto encode_rows_per_component(size_t data_size, size_t num_subsets)
{
    if (num_subsets <= 1)
        return 0.0;
    else
    {
        return log2_nchoosek(data_size - 1, num_subsets - 1);
    }
}

template <typename S>
double encode_which_rows_per_component(const S& data)
{
    size_t k = data.num_components();
    size_t n = data.size();
    if (k == 1)
        return 0;

    double l = 0;
    for (size_t i = 0; i < k; ++i)
    {
        auto n_i = data.subset(i).size();
        if (n_i == 0)
            continue;
        l += n_i * std::log2(double(n_i) / n);
    }
    return -l;
}

template <typename T = double, typename S>
std::pair<T, size_t>
encode_pattern_by_singletons(const S& x, size_t data_size, sd::slice<const T> fr)
{
    size_t length = 0;
    T      acc    = 0;
    // each item in pattern:
    foreach(x, [&](size_t item) {
        T s = fr[item] * data_size;
        T a = s == 0;
        acc -= std::log2((s + a) / (data_size + a)); // encode which item in pattern
        ++length;
    });
    return {acc, length};
}

template <typename T = double, typename S, typename Fr>
std::pair<T, size_t>
encode_pattern_by_singletons(const S& x, size_t data_size, size_t index, const Fr& fr)
{
    size_t length = 0;
    T      acc    = 0;
    // each item in pattern:
    foreach(x, [&](size_t item) {
        T s = fr(item, index) * data_size;
        T a = s == 0;
        acc -= std::log2((s + a) / (data_size + a)); // encode which item in pattern
        ++length;
    });
    return {acc, length};
}

template <typename Trait>
auto encode_summaries_expensive(const Composition<Trait>& c)
{
    using float_type = typename Trait::float_type;
    float_type acc   = 0;
    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        const auto n_i = c.data.subset(i).size();
        acc += std::log2(c.assignment[i].size()) * std::log2(n_i);
        for (auto j : c.assignment[i])
        {
            size_t      supp = c.frequency(j, i) * n_i;
            const auto& x    = c.summary.point(j);
            auto [e, l]      = encode_pattern_by_singletons<float_type>(x, n_i, i, c.frequency);

            acc += e + std::log2(l) + std::log2(supp + (supp == 0));
        }
    }
    return acc;
}

template <typename Trait>
auto encode_per_component_supports(const Composition<Trait>& c)
{
    typename Trait::float_type acc = 0;
    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        assert(c.assignment[i].size() >= c.data.dim);

        const auto n_i = c.data.subset(i).size();
        acc += universal_code(c.assignment[i].size()); // how many patterns
        for (auto j : c.assignment[i])
        {
            size_t s = c.frequency(j, i) * n_i;
            acc += universal_code(s);
        }
    }
    return acc;
}

auto encode_assignment_matrix_without_singletons(AssignmentMatrix const& am,
                                                 size_t                  summary_size,
                                                 size_t                  dim)
{
    const auto k   = am.size();
    const auto n_1 = trace(am) - k * dim;
    return log2_nchoosek(k * (summary_size - dim), n_1) + universal_code(n_1);
}

template <typename S, typename T>
T encode_patterns_once_globally(LabeledDataset<T, S> const& summary, size_t data_size)
{
    T acc = 0;
    for (const auto& [y, x] : summary)
    {
        if (is_singleton(x))
            continue;
        auto [l_j, length] = encode_pattern_by_singletons<T>(x, data_size, summary.labels());
        acc += l_j + (length > 1) * universal_code(length);
    }

    return acc;
}

template <typename Trait>
auto encode_model_mdl(const Composition<Trait>& c)
{
    typename Trait::float_type lm = 0;

#if 0
    lm += encode_summaries_expensive(c);
    lm += encode_num_component(c.assignment.size());
    lm += encode_rows_per_component(c.data.size(), c.assignment.size());
#else
    lm += encode_num_component(c.assignment.size());
    lm += encode_rows_per_component(c.data.size(), c.assignment.size());

    lm += encode_patterns_once_globally(c.summary, c.data.size());
    lm +=
        encode_assignment_matrix_without_singletons(c.assignment, c.summary.size(), c.data.dim);
    // lm += encode_per_component_supports(c);

#endif
    return lm;
}

template <typename Summary>
auto encode_model_mdl(const Summary& summary, size_t data_size)
{
    auto acc = encode_patterns_once_globally(summary, data_size);
    for (auto [fr, _] : summary)
    {
        acc += universal_code(static_cast<size_t>(fr * data_size));
    }
    return acc;
}

template <typename Trait>
auto encode_model_mdl(const Component<Trait>& c)
{
    return encode_model_mdl(c.summary, c.data.size());
}

template <typename C, typename X>
auto constant_mdl_cost(const C& c, const X& x)
{
    auto [l, length] = mdl::encode_pattern_by_singletons<typename C::float_type>(
        x, c.data.size(), c.summary.labels());
    l += mdl::universal_code(length);
    return l;
}
auto additional_cost_mdl(size_t support)
{
    return mdl::universal_code(support); // encode-per-component-support
}

template <typename Trait, typename Candidate, typename Masks>
auto desc_heuristic_mdl_multi(const Composition<Trait>& c,
                          const Candidate&          x,
                          const Masks&              masks,
                          const Config&             cfg)
{
    using float_type = typename Trait::float_type;

    float_type acc = -constant_mdl_cost(c, x.pattern);

    for (size_t i = 0; i < c.data.num_components(); ++i)
    {
        auto p = c.models[i].expectation(x.pattern);
        auto s = size_of_intersection(x.row_ids, masks[i]);
        auto q = static_cast<float_type>(s) / c.data.subset(i).size();
        auto h = s == 0 ? 0 : s * std::log2(q / p);

        assert(0 <= p && p <= 1);

        acc += h - additional_cost_mdl(s, c.models[i]);
    }

    return acc;
}

template <typename C, typename Distribution, typename Candidate>
auto desc_heuristic_mdl_1(const C& c, const Distribution& pr, const Candidate& x, const Config& cfg)
{
    using float_type = typename C::float_type;

    const auto s = static_cast<float_type>(x.support);
    const auto q = s / c.data.size();
    const auto p = pr.expectation(x.pattern);

    assert(0 <= p && p <= 1);

    return s * std::log2(q / p) - constant_mdl_cost(c, x.pattern);
}

} // namespace sd::disc::mdl