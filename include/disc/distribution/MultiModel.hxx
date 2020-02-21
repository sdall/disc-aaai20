#pragma once

#include <disc/distribution/Transactions.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/storage/Itemset.hxx>

#include <cstddef>
#include <numeric>
#include <vector>

#include <nonstd/optional.hpp>

namespace sd
{
namespace viva
{

// template <typename U, typename V>
// struct ItemsetObserver
// {
//     using float_type = V;

//     bool        known() const { return is_known; }
//     auto        frequency() const { return fr; }
//     const auto& point() const { return *buf; }
//     auto&       coeff() { return *theta; }
//     auto&       coeff0() { return *theta0; }

//     bool                    is_known;
//     float_type              fr;
//     const disc::itemset<U>* buf;
//     float_type*             theta0;
//     float_type*             theta;
// };

template <typename U, typename V>
struct SingletonModel
{
    using float_type   = V;
    using pattern_type = U;

    using index_type = std::size_t;

    struct singleton_storage
    {
        index_type element;
        float_type frequency;
        float_type theta       = 1;
        float_type probability = 0.5;
    };

    std::vector<singleton_storage> set;

    float_type       theta0 = 1;
    size_t           dim    = 0;
    disc::itemset<U> buffer;

    // ItemsetObserver<U, V> get(size_t i)
    // {
    //     assert(i < set.size());
    //     auto& s = set[i];
    //     buffer.clear();
    //     buffer.reserve(dim);
    //     buffer.insert(s.element);
    //     return {false, s.frequency, &buffer, &theta0, &s.theta};
    // }
    auto&       coefficient(size_t i) { return set[i].theta; }
    auto&       normalizer() { return theta0; }
    const auto& normalizer() const { return theta0; }
    auto&       probability(size_t i) { return set[i].probability; }
    const auto& frequency(size_t i) const { return set[i].frequency; }
    const auto& probability(size_t i) const { return set[i].probability; }
    const auto& coefficient(size_t i) const { return set[i].theta; }
    const auto& point(size_t i)
    {
        buffer.clear();
        buffer.insert(set[i].element);
        return buffer;
    }

    void insert(float_type label, index_type element)
    {
        auto it = std::find_if(
            set.begin(), set.end(), [element](const auto& i) { return i.element == element; });
        if (it != set.end())
        {
            it->frequency = label;
        }
        else
        {
            set.push_back({element, label});
        }
    }

    template <typename T>
    void insert(float_type label, const T& t)
    {
        set.push_back({static_cast<index_type>(front(t)), label});
    }

    size_t dimension() const { return dim; }
    size_t size() const { return set.size(); }

    template <typename Pattern_Type>
    nonstd::optional<float_type> get_precomputed_expectation(const Pattern_Type& x) const
    {
        auto element = front(x);
        auto it      = std::find_if(
            begin(set), end(set), [element](const auto& x) { return x.element == element; });
        if (it != end(set))
        {
            return {it->probability};
        }
        else
            return nonstd::nullopt;
    }
};

template <typename U, typename V>
struct ItemsetStorage
{
    using float_type = V;

    disc::itemset<U> point{};
    float_type       frequency   = 0;
    float_type       theta       = 1;
    float_type       probability = 0.5;
};

template <typename U, typename V>
struct ItemsetModel
{
    using float_type   = V;
    using pattern_type = U;

    using itemset_storage = ItemsetStorage<U, V>;

    std::vector<itemset_storage> set;

    float_type       theta0         = 1;
    size_t           dim            = 0;
    size_t           num_singletons = 1;
    disc::itemset<U> buffer;
    std::vector<Block<U, V>> partitions;

    

    //     thread_local std::vector<Block<S, T>> transactions_x;
    // compute_counts(dimension_of_factor(model, x), model.itemsets, transactions_x);
    // // compute_counts(dimension_of_factor(model, x), augment_model(model, x), transactions_x);
    // return expected_frequency_known(transactions_x, model, x) ;


    // ItemsetObserver<U, V> get(size_t i)
    // {
    //     assert(i < size());
    //     auto& s = set[i];
    //     return {true, s.frequency, &s.point, &theta0, &s.theta};
    // }

    template <typename T>
    void insert(float_type label, const T& t)
    {
        buffer.clear();
        buffer.reserve(dim);
        buffer.insert(t);
        auto it = std::find_if(set.begin(), set.end(), [&](const auto& i) {
                return equal(i.point, buffer);
            });
        if (it != set.end())
        {
            it->frequency = label;
        }
        else
        {
            set.push_back({buffer, label, 1});
            // update_partitions();
        }
    }

    void update_partitions() 
    {
        compute_counts(width(), *this, partitions);
    }

    size_t width() const { return num_singletons; } 
    size_t dimension() const { return dim; }
    size_t size() const { return set.size(); }

    auto&       coefficient(size_t i) { return set[i].theta; }
    auto&       normalizer() { return theta0; }
    const auto& normalizer() const { return theta0; }
    auto&       probability(size_t i) { return set[i].probability; }
    const auto& frequency(size_t i) const { return set[i].frequency; }
    const auto& probability(size_t i) const { return set[i].probability; }
    const auto& point(size_t i) const { return set[i].point; }
    const auto& coefficient(size_t i) const { return set[i].theta; }

    decltype(auto) operator[](size_t i) const { return point(i); }
    bool           empty() const { return set.empty(); }

    template <typename Pattern_Type>
    nonstd::optional<float_type> get_precomputed_expectation(const Pattern_Type& x) const
    {
        const auto it = std::find_if(
            set.begin(), set.end(), [&](const auto& s) { return equal(s.point, x); });
        if (it != set.end())
        {
            return {it->probability};
        }
        else
            return nonstd::nullopt;
    }
};

template <typename U, typename V>
struct MultiModel
{
    using float_type   = V;
    using pattern_type = U;

    SingletonModel<U, V> singletons;
    ItemsetModel<U, V>   itemsets;

    explicit MultiModel(size_t w = 0)
    {
        singletons.dim          = w;
        itemsets.dim            = w;
        itemsets.num_singletons = 1;
    }

    auto& coefficient(size_t i) const
    {
        return i < singletons.set.size() ? singletons.coefficient(i)
                                         : itemsets.coefficient(i - singletons.set.size());
    }
    auto& coefficient(size_t i)
    {
        return i < singletons.set.size() ? singletons.coefficient(i)
                                         : itemsets.coefficient(i - singletons.set.size());
    }
    auto&       normalizer() { return itemsets.normalizer(); }
    const auto& normalizer() const { return itemsets.normalizer(); }
    auto        frequency(size_t i) const
    {
        return i < singletons.set.size() ? singletons.frequency(i)
                                         : itemsets.frequency(i - singletons.set.size());
    }
    auto& probability(size_t i)
    {
        return i < singletons.set.size() ? singletons.probability(i)
                                         : itemsets.probability(i - singletons.set.size());
    }
    auto& probability(size_t i) const
    {
        return i < singletons.set.size() ? singletons.probability(i)
                                         : itemsets.probability(i - singletons.set.size());
    }
    const auto& point(size_t i)
    {
        return i < singletons.set.size() ? singletons.point(i)
                                         : itemsets.point(i - singletons.set.size());
    }
    bool   is_pattern_known(size_t i) const { return i >= singletons.set.size(); }
    size_t size() const { return singletons.set.size() + itemsets.set.size(); }
    size_t dimension() const { return singletons.dim; }

    template <typename T>
    bool is_pattern_feasible(const T&) const
    {
        return true;
    }

    template <typename T>
    void insert_pattern(float_type label, const T& t, bool estimate = false)
    {
        itemsets.insert(label, t);
        itemsets.num_singletons = singletons.set.size();
        if (estimate) {
            estimate_model(*this);
        }    
    }

    template <typename T>
    void insert_singleton(float_type label, T&& t, bool estimate = false)
    {
        singletons.insert(label, t);
        itemsets.num_singletons = singletons.set.size();
        if (estimate) {
            estimate_model(*this);
        }
    }

    template <typename T>
    void insert(float_type label, const T& t, bool estimate = false)
    {
        if (is_singleton(t))
        {
            insert_singleton(label, t);
        }
        else
        {
            insert_pattern(label, t);
        }

        if (estimate)
            estimate_model(*this);
    }

    template <typename Pattern_Type>
    nonstd::optional<float_type> get_precomputed_expectation(const Pattern_Type& x) const
    {
        return is_singleton(x) ? singletons.get_precomputed_expectation(x)
                               : itemsets.get_precomputed_expectation(x);
    }
};

template <typename S, typename T, typename U>
bool contains_pattern(const MultiModel<S, T>& m, const U& t)
{
    return std::any_of(
        m.itemsets.set.begin(), m.itemsets.set.end(), [&, n = count(t)](const auto& i) {
            const auto o = count(i.point);
            return o == n && is_subset(i.point, t);
        });
}

} // namespace viva
} // namespace sd