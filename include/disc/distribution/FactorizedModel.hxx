#pragma once

#include <disc/distribution/MultiModel.hxx>
#include <disc/storage/Itemset.hxx>
// #include <disc/distribution/Cache.hxx>

#ifndef NDEBUG
#include <exception>
#endif

namespace sd
{
namespace viva
{

template <typename U, typename V, typename Underlying_Factor_Type = MultiModel<U, V>>
struct FactorizedModel
{
    using float_type            = V;
    using value_type            = V;
    using pattern_type          = U;
    using index_type            = std::size_t;
    using underlying_model_type = Underlying_Factor_Type;
    // using cache_type            = Cache<U, V>;

    struct factor_type
    {
        explicit factor_type(size_t dim) : factor(dim) { range.reserve(dim); }
        sd::disc::itemset<U>  range;
        underlying_model_type factor;
    };

    explicit FactorizedModel(size_t dim) : dim(dim) { init(dim); }

    size_t dim              = 0;
    size_t total_size       = 0;
    size_t max_num_itemsets = 5;
    size_t max_range_size   = 8;

    std::vector<factor_type> factors;

    size_t size() const { return total_size; }
    size_t dimension() const { return dim; }

    void init(size_t dim)
    {
        this->dim = dim;
        factors.clear();
        factors.resize(dim, factor_type(dim));

#pragma omp parallel for
        for (size_t i = 0; i < dim; ++i)
        {
            factors[i].range.insert(i);
            factors[i].factor.insert_singleton(0.5, i);
            estimate_model(factors[i].factor);
        }
    }

    void clear() { factors.clear(); }

    void insert_singleton(value_type frequency, const index_type element, bool estimate = false)
    {
        size_t index = 0;
        for (auto& f : factors)
        {
            if (is_subset(element, f.range))
            {
                f.factor.insert_singleton(frequency, element, estimate);
                return;
            }
            index++;
        }
        auto& f = factors.emplace_back(dim);
        f.range.insert(element);
        f.factor.insert_singleton(frequency, element, estimate);
    }

    template <typename T>
    void insert_singleton(value_type frequency, const T& t, bool estimate = false)
    {
        insert_singleton(frequency, static_cast<index_type>(front(t)), estimate);
    }

    // void insert_singleton(const size_t element) { insert_singleton(0.5, element); }

    template <typename T>
    void insert_pattern(value_type frequency,
                        const T&   t,
                        size_t     max_num_itemsets,
                        size_t     /*max_range_size*/,
                        bool       estimate)
    {
        std::vector<size_t> selection;
        selection.reserve(count(t));
        
        for (size_t i = 0, n = factors.size(); i < n; ++i)
        {
            auto& f = factors[i];
            if (is_subset(t, f.range))
            {
                if (f.factor.itemsets.set.size() < max_num_itemsets)
                {
                    f.factor.insert(frequency, t, estimate);
                }
                return;
            }
            if (intersects(t, f.range))
                selection.push_back(i);
        }

        if(selection.empty()) {
            return;
        }

        factor_type next(dim);

        for (const auto& s : selection)
        {
            join_factors(next, factors[s]);
        }
        next.factor.insert(frequency, t);

        if(estimate)
            estimate_model(next.factor);

//         if (count(next.range) > max_range_size ||
//             next.factor.itemsets.set.size() > max_num_itemsets)
//         {
// #ifndef NDEBUG
//             struct patterns_not_feasible_exception : std::domain_error
//             {
//                 patterns_not_feasible_exception(const char* c) : std::domain_error(c) {}
//             };
//             throw patterns_not_feasible_exception{"pattern too large or factor is full"};
// #endif
//             return;
//         }

        for (auto i : selection)
        {
            factors[i].range.clear();
        }
        selection.clear();

        erase_empty_factors();

        factors.emplace_back(std::move(next));
    }

    template <typename T>
    void insert_pattern(value_type frequency, const T& t, bool estimate = false)
    {
        insert_pattern(frequency, t, max_num_itemsets, max_range_size, estimate);
    }

    template <typename T>
    void insert(value_type frequency, const T& t)
    {
        if (is_singleton(t))
            insert_singleton(frequency, t);
        else
            insert_pattern(frequency, t);
    }

    template <typename T>
    void insert(value_type frequency, const T& t, bool estimate)
    {
        if (is_singleton(t))
            insert_singleton(frequency, t, estimate);
        else
            insert_pattern(frequency, t, estimate);
    }

    template <typename T>
    bool is_pattern_feasible(const T& t, size_t max_num_itemsets, size_t max_range_size) const
    {
        size_t total_size  = 0;
        size_t total_width = 0;
        for (auto& f : factors)
        {
            if (is_subset(t, f.range))
            {
                return f.factor.itemsets.set.size() < max_num_itemsets;
            }
            if (intersects(t, f.range))
            {
                // this works, because all factors are covering disjoint sets
                total_size += f.factor.itemsets.set.size();
                total_width += count(f.range);
                if (total_size >= max_num_itemsets || total_width >= max_range_size)
                    return false;
            }
        }

        return true;
    }

    template <typename T>
    bool is_pattern_feasible(const T& t) const
    {
        return is_pattern_feasible(t, max_num_itemsets, max_range_size);
    }

    static void join_factors(factor_type& f, const factor_type& g)
    {
        for (auto& t : g.factor.itemsets.set)
            f.factor.insert(t.frequency, t.point);

        for (auto& t : g.factor.singletons.set)
            f.factor.insert_singleton(t.frequency, t.element);

        f.range.insert(g.range);
    }

    underlying_model_type as_single_factor() const
    {
        factor_type next(dim);
        for (const auto& s : factors)
        {
            join_factors(next, s);
        }
        return next.factor;
    }

    void erase_empty_factors()
    {
        for (size_t i = 0; i < factors.size();)
        {
            if (factors[i].range.empty())
            {
                std::swap(factors[i], factors.back());
                factors.pop_back();
            }
            else
            {
                ++i;
            }
        }
    }
};

} // namespace viva
} // namespace sd