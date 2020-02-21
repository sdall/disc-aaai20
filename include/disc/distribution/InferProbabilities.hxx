#pragma once

#include <disc/distribution/FactorizedModel.hxx>
#include <disc/distribution/MultiModel.hxx>
#include <disc/distribution/Transactions.hxx>
#include <disc/storage/Itemset.hxx>
#include <disc/utilities/SizeOfUnion.hxx>
#include <disc/utilities/Support.hxx>

#include <nonstd/optional.hpp>

#include <numeric>

namespace sd
{
namespace viva
{

template <typename pattern_type, typename float_type, typename query_type>
auto probability_transaction_2(MultiModel<pattern_type, float_type> const& c,
                             const query_type&                             t)
{

    thread_local disc::itemset<pattern_type> xx;
    xx.clear();
    xx.insert(t);

    float_type acc = c.itemsets.theta0;
    for (size_t i = 0, l = c.itemsets.set.size(); i < l; ++i)
    {
        setminus(xx, c.itemsets.set[i].point);
        if (is_subset(c.itemsets.set[i].point, t))
        {
            acc *= c.itemsets.set[i].theta;
        }
    }
    for (size_t i = 0, l = c.singletons.set.size(); i < l; ++i)
    {
        if (is_subset(c.singletons.set[i].element, xx))
        {
            acc *= c.singletons.set[i].theta;
        }
    }
    
    return acc;
}


template <typename pattern_type, typename float_type, typename query_type>
auto probability_transaction(ItemsetModel<pattern_type, float_type> const& c,
                             const query_type&                             t)
{
    float_type acc = c.theta0;
    for (size_t i = 0, l = c.set.size(); i < l; ++i)
    {
        if (is_subset(c.set[i].point, t))
        {
            acc *= c.set[i].theta;
        }
    }
    return acc;
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability_transaction(SingletonModel<pattern_type, float_type> const& c,
                             query_type const&                               t)
{
    float_type acc = c.theta0;
    // if (!is_singleton(t)) return acc;
    for (size_t i = 0, l = c.set.size(); i < l; ++i)
    {
        if (is_subset(c.set[i].element, t))
        {
            acc *= c.set[i].theta;
        }
    }
    return acc;
}

template <typename pattern_type, typename float_type, typename query_type>
auto probability_transaction(MultiModel<pattern_type, float_type> const& c, query_type const& t)
{
    return probability_transaction(c.itemsets, t) * probability_transaction(c.singletons, t);
}

template <typename pattern_type,
          typename float_type,
          typename underlying_factor,
          typename query_type>
auto probability_transaction(
    FactorizedModel<pattern_type, float_type, underlying_factor> const& c, query_type const& t)
{
    return std::accumulate(
        c.factors.begin(), c.factors.end(), float_type(1), [&t](auto acc, const auto& f) {
            return acc * probability_transaction(f.factor, t);
        });
}

template <typename Transactions, typename Model, typename Pattern>
auto expected_frequency_known(Transactions const& transactions,
                              Model const&        model,
                              Pattern const&      x)
{
    using float_type = typename Model::float_type;
    float_type p     = 0;

    for (auto const& t : transactions)
    {
        if (t.value != 0 && is_subset(x, t.cover))
        {
            // p += probability_transaction(model, t.cover);
            p += t.value * probability_transaction(model, t.cover);
            assert(!std::isnan(p));
            assert(!std::isinf(p));
        }
    }
    return p;
}



template <typename Model_Type>
size_t dimension_of_factor(const Model_Type& m)
{
    return m.singletons.size();
    // return m.dimension();
}
template <typename Model_Type, typename T>
size_t dimension_of_factor(const Model_Type& m, const T&)
{
    return m.singletons.size();
    // return m.dimension();
}

template <typename Pattern_Type, typename Float_type>
struct AugmentedModel
{
    using pattern_type = Pattern_Type;
    using float_type   = Float_type;

    AugmentedModel(ItemsetModel<pattern_type, float_type> const& underlying_model,
                   disc::itemset<pattern_type> const&            additional_pattern)
        : underlying_model(underlying_model), additional_pattern(additional_pattern)
    {
    }

    size_t size() const { return underlying_model.size() + 1; }

    const auto& point(size_t i) const
    {
        if (i < underlying_model.size())
        {
            return underlying_model.point(i);
        }
        else
        {
            return additional_pattern;
        }
    }

    ItemsetModel<pattern_type, float_type> const& underlying_model;
    disc::itemset<pattern_type> const&            additional_pattern;
};

template <typename S, typename T>
auto augment_model(MultiModel<S, T> const& model, disc::itemset<S> const& x)
{
    return AugmentedModel<S, T>(model.itemsets, x);
}

template <typename S, typename T>
auto expected_frequency_unknown(MultiModel<S, T> const& model, disc::itemset<S> const& x)
{
    thread_local std::vector<Block<S, T>> transactions_x;
    viva::compute_counts(dimension_of_factor(model, x), augment_model(model, x), transactions_x);
    return expected_frequency_known(transactions_x, model, x) ;

    // return expected_frequency_known(model.itemsets.partitions, model, x) ; 
}



template <typename S, typename T, typename Blocks>
void compute_transactions(MultiModel<S, T> const& model,
                          disc::itemset<S> const& x,
                          bool                    known,
                          Blocks&                 blocks)
{
    if (!known)
    {
        viva::compute_counts(dimension_of_factor(model, x), augment_model(model, x), blocks);
    }
    else
    {
        viva::compute_counts(dimension_of_factor(model, x), model.itemsets, blocks);
    }
}

template <typename S, typename T>
auto expected_frequency(MultiModel<S, T> const& model, disc::itemset<S> const& x)
{
    if (auto p = model.get_precomputed_expectation(x); p)// && !is_singleton(x))
    {
        // return expected_frequency_known(model.itemsets.partitions, model, x) ; 
        return p.value();
    }
    else
    {
        return expected_frequency_unknown(model, x);
    }
}

template <typename Model, typename query_type>
auto probability_of_absent_items(Model const& m, query_type const& t)
{
    using float_type = typename Model::float_type;
    float_type p     = 1;
    for (const auto& s : m.singletons.set)
    {
        if (!is_subset(s.element, t))
        {
            p *= float_type(1.0) - s.probability;
        }
    }
    return p;
}

template <typename Model, typename query_type>
auto expected_generalized_frequency(Model const& m, query_type const& t)
{
    return expected_frequency(m, t) * probability_of_absent_items(m, t);
}

template <typename pattern_type, typename float_type, typename Underlying, typename query_type>
auto expected_frequency(const FactorizedModel<pattern_type, float_type, Underlying>& fm,
                        const query_type&                                            t)
{
    thread_local disc::itemset<pattern_type> part;
    part.clear();
    part.reserve(fm.dim);

    float_type estimate = 1;

    for (size_t i = 0; i < fm.factors.size(); ++i)
    {
        const auto& f = fm.factors[i];
        if (intersects(t, f.range))
        {
            intersection(t, f.range, part);
            estimate *= expected_frequency(f.factor, part);
        }
    }
    // assert(!std::isnan(estimate) && estimate != 0 && !std::isinf(estimate));
    return estimate;
}

template <typename query_type, typename pattern_type, typename float_type, typename Underlying>
auto expected_generalized_frequency(
    const FactorizedModel<pattern_type, float_type, Underlying>& m, const query_type& t)
{
    float_type p = expected_frequency(m, t);
    for (const auto& f : m.factors)
    {
        p *= probability_of_absent_items(f.factor, t);
    }
    return p;
}

template <typename S, typename pattern_type, typename float_type>
auto infer_itemset_frequency_approx(const S&                                         t,
                                    const FactorizedModel<pattern_type, float_type>& fm)
{
    thread_local disc::itemset<pattern_type> part;
    part.clear();
    part.reserve(fm.dim);

    float_type estimate = 1;
    for (size_t i = 0; i < fm.factors.size(); ++i)
    {
        const auto& f = fm.factors[i];
        if (intersects(t, f.range))
        {
            intersection(t, f.range, part);

            if (auto p = f.factor.get_precomputed_expectation(part); p)
            {
                estimate *= p.value();
            }
            else
            {
                for (const auto& s : f.singletons.set)
                {
                    if (is_subset(s.element, part))
                        estimate *= s.probability;
                }
            }
        }
    }
    return estimate;
}

} // namespace viva
} // namespace sd
