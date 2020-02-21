#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/distribution/Distribution.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/utilities/Support.hxx>
#include <disc/utilities/UniversalIntEncoding.hxx>

#include <math/nchoosek.hxx>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sd
{
namespace disc
{

template <typename S, typename T, typename Data_Type>
auto encode_data_mtv_quick_impl(const viva::MultiModel<S, T>& model, const Data_Type& data)
{
    T h = -std::log2(model.normalizer());
    for (size_t i = 0; i < model.size(); ++i)
    {
        h -= model.frequency(i) * std::log2(model.coefficient(i));
    }

    // for (size_t i = 0; i < model.itemsets.set.size(); ++i)
    // {
    //     h -= model.itemsets.frequency(i) * std::log2(model.itemsets.coefficient(i));
    // }
    // h -= std::log2(model.singletons.normalizer());
    // for (size_t i = 0; i < model.singletons.set.size(); ++i)
    // {
    //     h -= (1 - model.singletons.frequency(i)) *
    //     std::log2(model.singletons.coefficient(i));
    // }

    for (size_t i = 0; i < model.singletons.set.size(); ++i)
    {
        const auto r = model.singletons.frequency(i);
        if (r == 0.0 || r == 1.0)
            continue;
        h += -r * std::log2(r) + (1 - r) * std::log2(1 - r);
    }

    return h * data.size();
}

template <typename S, typename T, typename Data_Type>
auto encode_data_mtv_quick_impl(const viva::FactorizedModel<S, T>& model, const Data_Type& data)
{
    using float_type = T;
    float_type h     = 0;
    for (const auto& f : model.factors)
    {
        const auto& m = f.factor;
        h += encode_data_mtv_quick_impl(m, data);
    }

    return h;
}

template <typename Distribution_Type, typename Data_Type>
auto encode_data_mtv(Distribution_Type const& model, const Data_Type& data)
{
    return encode_data_mtv_quick_impl(model.model, data);
}

// template <typename Distribution_Type, typename Data_Type>
// auto encode_data_duplicated_data(Distribution_Type const& model, const Data_Type& data)
// {
//     // return encode_data_mtv(model, data);
//     using value_type = typename Distribution_Type::value_type;
//     using pattern_type = typename Data_Type::pattern_type;

//     assert(!data.empty());

//     // resillience against unlikely numeric issues using
//     // laplace smoothing
//     const auto l = value_type(1) / (model.model.dim + data.size());
//     const auto u = value_type(1) - l;

//     struct Equal {
//         template <typename S> bool operator()(const S& a, const S& b) {
//             return sd::equal(a, b);
//         }
//     };

//     struct Hash {
//         bool operator()(const itemset<tag_sparse>& a) const {
//             size_t h = 0;
//             iterate_over(a, [](size_t j) {
//                 h ^= std::hash<size_t>(j);
//             });
//             return h;
//         }
//         bool operator()(const itemset<tag_dense>& a) const {
//             size_t h = 0;
//             for(const auto j : a.container())
//                 h ^= std::hash<size_t>(j);
//             });
//             return h;
//         }
//     };

//     std::unordered_map<itemset<pattern_type>, value_type, Hash, Equal> cache;

//     value_type acc = 0;
// #pragma omp parallel for reduction(+ : acc) private(cache)
//     for (size_t i = 0; i < data.size(); ++i)
//     {
//         const auto& x = point(data[i]);
//         auto it = cache.find(x);
//         value_type p = 1;
//         if(it != cache.end())
//         {
//             p = *it;
//         }
//         else {
//             p = model.expected_generalized_frequency(x);
//             cache.insert(x, p);
//         }

//         assert(!std::isinf(p));
//         assert(!std::isnan(p));
//         assert(p > 0);
//         assert(p <= 1);

//         std::clamp(p, l, u);
//         acc -= std::log2(p);
//     }
//     return acc;
// }

template <typename Distribution_Type, typename Data_Type>
auto encode_data(Distribution_Type const& model, const Data_Type& data)
{
    // return encode_data_mtv(model, data);
    using float_type = typename Distribution_Type::float_type;

    assert(!data.empty());

    // resillience against unlikely numeric issues using
    // laplace smoothing
    const auto l = float_type(1) / (model.model.dimension() + data.size());
    const auto u = float_type(1) - l;

    float_type acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (size_t i = 0; i < data.size(); ++i)
    {
        auto p = model.expected_generalized_frequency(point(data[i]));

        assert(!std::isinf(p));
        assert(!std::isnan(p));
        assert(p > 0);
        assert(p <= 1);

        std::clamp(p, l, u);
        acc -= std::log2(p);
    }
    return acc;
}

template <typename distribution_type, typename data_type>
auto encode_subsets(const std::vector<distribution_type>& s, const data_type& data)
{
    using value_type = typename distribution_type::value_type;
    value_type l     = 0;
    for (size_t i = 0; i < s.size(); ++i)
    {
        l += encode_data(s[i], data.subset(i));
    }
    return l;
}

template <typename distribution_type, typename data_type, typename value_type>
auto encode_subsets(const std::vector<distribution_type>& s,
                    const data_type&                      data,
                    std::vector<value_type>&              individual_encoding_lenghts)
{
    assert(individual_encoding_lenghts.size() == s.size());
    assert(data.num_components() == s.size());

    value_type l = 0;
    for (size_t i = 0; i < data.num_components(); ++i)
    {
        individual_encoding_lenghts[i] = encode_data(s[i], data.subset(i));
        l += individual_encoding_lenghts[i];
    }
    return l;
}

template <typename Trait, typename D, typename T>
void fill_data_encodings(Composition<Trait> const& c,
                         std::vector<D> const&     models,
                         std::vector<T>&           encodings)
{
    assert(encodings.size() == c.assignment.size());
    assert(models.size() == c.assignment.size());
    assert(c.data.num_components() == c.assignment.size());

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        encodings[i] = disc::encode_data(models[i], c.data.subset(i));
    }
}

template <typename Trait, typename D>
auto make_data_encodings(const Composition<Trait>& c, const std::vector<D>& models)
{
    using T = typename Trait::float_type;
    std::vector<T> encodings(c.assignment.size());
    fill_data_encodings(c, models, encodings);
    return encodings;
}

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
        return math::log2_nchoosek(data_size - 1, num_subsets - 1);
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

// template <typename T = double, typename S, typename Data>
// std::pair<T, size_t> encode_pattern_by_singletons(const S& x, const Data& data)
// {
//     size_t length = 0;
//     T      acc    = 0;
//     // each item in pattern:
//     iterate_over(x, [&](size_t item) {
//         T s = support(item, data);
//         T a = s == 0;
//         acc -= std::log2((s + a) / (data.size() + a)); // encode which item in pattern
//         ++length;
//     });
//     return {acc, length};
// }

template <typename T = double, typename S>
std::pair<T, size_t>
encode_pattern_by_singletons(const S& x, size_t data_size, cpslice<const T> fr)
{
    size_t length = 0;
    T      acc    = 0;
    // each item in pattern:
    iterate_over(x, [&](size_t item) {
        T s = fr[item] * data_size;
        T a = s == 0;
        acc -= std::log2((s + a) / (data_size + a)); // encode which item in pattern
        ++length;
    });
    return {acc, length};
}

template <typename T, typename S>
std::pair<T, size_t> encode_pattern_by_singletons(const S&                  x,
                                                  const cpslice<const T, 2> fr,
                                                  size_t                    component_index,
                                                  size_t                    data_size)
{
    size_t length = 0;
    T      acc    = 0;
    // each item in pattern:
    iterate_over(x, [&](size_t item) {
        auto s = fr(item, component_index) * data_size;
        T    a = s == 0;
        acc -= std::log2(T(s + a) / (data_size + a)); // encode which item in pattern
        ++length;
    });
    return {acc, length};
}

template <typename Trait>
struct PatternsetResult;

template <typename Trait, typename U>
auto additional_cost_mdl(const PatternsetResult<Trait>& c, const U& x, size_t support)
{
    using float_type = typename Trait::float_type;

    auto [enc, length] =
        encode_pattern_by_singletons<float_type>(x, c.data.size(), c.summary.template col<0>());
    return enc + std::log2(length) + std::log2(support + (support == size_t(0)));
}

template <typename Trait, typename U>
auto additional_cost_mdl_expensive(const Composition<Trait>& c,
                                   const U&                  x,
                                   size_t                    support,
                                   size_t                    component_index,
                                   size_t                    component_size)
{
    using float_type   = typename Trait::float_type;
    auto [enc, length] = encode_pattern_by_singletons<float_type>(
        x, c.frequency, component_index, component_size);
    return enc + std::log2(length) + std::log2(support + (support == 0));
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
            size_t support = c.frequency(j, i) * n_i;
            acc += additional_cost_mdl_expensive(c, c.summary.point(j), support, i, n_i);
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

template <typename Trait>
auto encode_per_component_supports_everything(const Composition<Trait>& c)
{
    typename Trait::float_type acc = 0;
    for (size_t j = 0; j < c.model.size(); ++j)
    {
        for (size_t i = 0; i < c.summary.size(); ++i)
        {
            const auto n_j = c.data.subset(j).size();
            acc += universal_code(n_j * c.frequency(i, j));
        }
    }
    return acc;
}

template <typename Trait>
auto encode_per_component_supports_diffset(const Composition<Trait>& c)
{
    const auto n      = c.data.size();
    const auto global = c.frequency.extent(1) - 1;

    typename Trait::float_type acc = 0;
    for (size_t j = 0; j < c.frequency.length(); ++j)
    {
        acc += universal_code(size_t(c.frequency(j, global) * n));
    }

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        const auto n_i = c.data.subset(i).size();
        acc += universal_code(c.assignment[i].size()); // how many patterns
        for (auto j : c.assignment[i])
        {
            size_t s = std::abs(c.frequency(j, i) * n_i - c.frequency(j, global) * n);
            acc += universal_code(s) + 1; // 1 bit for the sign
        }
    }
    return acc;
}

template <typename S, typename T>
T encode_patterns_once_globally(LabeledDataset<T, S> const& summary, size_t data_size)
{
    T acc = 0;
    for (const auto& [y, x] : summary)
    {
        if (is_singleton(x))
            continue;
        auto [l_j, length] =
            encode_pattern_by_singletons<T>(x, data_size, summary.template col<0>());
        acc += l_j + (length > 1) * universal_code(length);
    }

    return acc;
}

template <typename Trait>
auto encode_patterns_once_globally(const Composition<Trait>& c)
{
    return encode_patterns_once_globally(c.summary, c.data.size());
}

auto encode_assignment_matrix_1bit_per(AssignmentMatrix const& am,
                                       size_t                  summary_size,
                                       size_t                  dim)
{
    return am.size() * (summary_size - dim);
}

auto encode_assignment_matrix_1bit_costly(AssignmentMatrix const& am,
                                          size_t                  summary_size,
                                          size_t)
{
    return am.size() * summary_size;
}

auto encode_assignment_matrix(AssignmentMatrix const& am, size_t summary_size, size_t)
{
    const auto k   = am.size();
    const auto n_1 = std::accumulate(
        begin(am), end(am), size_t(), [](auto acc, const auto& x) { return acc + x.size(); });
    return math::log2_nchoosek(k * summary_size, n_1) + universal_code(n_1);
}

auto encode_assignment_matrix_without_singletons(AssignmentMatrix const& am,
                                                 size_t                  summary_size,
                                                 size_t                  dim)
{
    const auto k = am.size();
    const auto n_1 =
        std::accumulate(begin(am), end(am), size_t(), [dim](auto acc, const auto& x) {
            return acc + x.size() - dim;
        });
    return math::log2_nchoosek(k * (summary_size - dim), n_1) + universal_code(n_1);
}

template <typename Trait>
auto encode_summaries_efficient(const Composition<Trait>& c)
{
    typename Trait::float_type acc = 0;
    acc += encode_patterns_once_globally(c);
    acc += encode_assignment_matrix_1bit_per(c.assignment, c.summary.size(), c.data.dim);
    acc += encode_per_component_supports(c);
    return acc;
}

template <typename Trait>
auto encode_model_mdl(const Composition<Trait>& c)
{
    typename Trait::float_type lm = 0;
#if 1
    lm += encode_summaries_expensive(c);
#else
    lm += encode_num_component(c.assignment.size());
    lm += encode_rows_per_component(c.data.size(), c.assignment.size());
    // lm += encode_which_rows_per_component(c.data);
    lm += encode_summaries_efficient(c);
#endif
    return lm;
}

template <typename Trait, typename U>
auto additional_cost_mdl(const Composition<Trait>& c,
                         const U&                  x,
                         size_t                    support,
                         size_t                    component_index,
                         size_t                    component_size)
{
#if 1
    return additional_cost_mdl_expensive(c, x, support, component_index, component_size);
#else
    T           lm     = 0;
    const auto& fr     = summary.template col<0>().container();
    auto [l_j, length] = encode_pattern_by_singletons<T>(x, c.data.size(), fr);
    lm += l_j;
    lm += universal_code(length);

    if constexpr (meta::has_components_member_fn<data_type>::value)
    {
        // encode_supports_per_component using average-cost of a universal code.
        // cost for assignment matrix most likely actually reduce
        lm += (c.data.num_components() - 1) * universal_code(4);
    }
    return lm;

#endif
}

template <typename Trait, typename U>
auto additional_cost_mdl_assignment(const Composition<Trait>& c,
                                    const U&                  x,
                                    size_t                    support,
                                    size_t                    component_index,
                                    size_t                    component_size)
{
    return additional_cost_mdl(c, x, support, component_index, component_size);
}

template <typename Trait>
auto encode_model_bic(const Composition<Trait>& c)
{
    const auto n = c.data.size();
    const auto s = c.summary.size();
    const auto k = c.data.num_components();
    const auto d = c.data.dim;
    const auto m = s - d;
    // const auto tr = std::accumulate(
    //     begin(c.assignment), end(c.assignment), size_t(), [&](auto acc, const auto& x) {
    //         return acc + count(x) - d;
    //     });
    // const auto df = k * m + tr + n; // n is constant

    const auto df = k * (2 * m + d) + n; // n is constant
    return (std::log2(n) * df) / 2;
}

template <typename Trait>
auto encode_model(const Composition<Trait>& c, const bool bic = false)
{
    return bic ? encode_model_bic(c) : encode_model_mdl(c);
}

template <typename S, typename T>
T encode_model_sdm(LabeledDataset<T, S> const& summary, size_t data_size, bool bic = false)
{
    if (bic)
    {
        return summary.size() * std::log2(data_size) / 2;
    }
    else
    {
        // return encode_summary_mtv<T>(summary, data_size, data_dim);
        T acc = 0;
        for (auto [fr, _] : summary)
        {
            size_t s = fr * data_size;
            acc += universal_code(s);
        }

        acc += encode_patterns_once_globally(summary, data_size);
        return acc;
    }
}

template <typename S, typename Distribution_Type, typename patternset_type>
auto encoding_length_sdm(Distribution_Type&      model,
                         const disc::Dataset<S>& data,
                         const patternset_type&  items,
                         bool                    bic = false)
{
    using float_type = typename Distribution_Type::float_type;
    return EncodingLength<float_type>{disc::encode_data(model, data),
                                      encode_model_sdm(items, data.size(), bic)};
}

template <typename Trait, typename T>
auto encoding_length_mdm_fill(const Composition<Trait>& c,
                              std::vector<T>&           subset_encodings,
                              const bool                bic)
{
    subset_encodings.resize(c.models.size());
    auto ld = encode_subsets(c.models, c.data, subset_encodings);
    auto lm = encode_model(c, bic);
    return EncodingLength<T>{ld, lm};
}

template <typename Trait>
auto encoding_length_mdm(Composition<Trait>& c, const bool bic)
{
    return c.encoding = encoding_length_mdm_fill(c, c.subset_encodings, bic);
}

template <typename Trait, typename T>
void data_encoding_length_mdm_update(const Composition<Trait>& c,
                                     EncodingLength<T>&        l,
                                     std::vector<T>&           subset_encodings,
                                     size_t                    index)
{
    assert(index < c.models.size());
    l.of_data -= subset_encodings[index];
    subset_encodings[index] = encode_data(c.models[index], c.data.subset(index));
    l.of_data += subset_encodings[index];
}

template <typename Trait>
auto encoding_length_mdm_update(Composition<Trait>&       c,
                                std::pair<size_t, size_t> index,
                                const bool                bic)
{
    data_encoding_length_mdm_update(c, c.encoding, c.subset_encodings, index.first);
    data_encoding_length_mdm_update(c, c.encoding, c.subset_encodings, index.second);
    c.encoding.of_model = encode_model(c, bic);
    return c.encoding;
}

namespace meta
{
template <typename T, typename = void>
struct has_components_member_fn : std::false_type
{
};

template <typename T>
struct has_components_member_fn<T, std::void_t<decltype(std::declval<T>().num_components())>>
    : std::true_type
{
};

} // namespace meta

template <typename C>
auto additional_cost_bic(const C& c) -> typename C::float_type
{
    using data_type = std::decay_t<decltype(c.data)>;

    size_t df = 1;
    if constexpr (meta::has_components_member_fn<data_type>::value)
    {
        auto k = c.data.num_components();
        df += k > 1 ? k : 0;
    }
    return std::log2(c.data.size()) * df / 2;
}

} // namespace disc
} // namespace sd