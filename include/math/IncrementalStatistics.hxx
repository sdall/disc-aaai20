#pragma once
#include <cmath>
#include <cstddef>
#include <functional> // less

// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

namespace sd
{

/** Incremental, simultaneous computation of mean and variance */
template <class val = double, class size_t = std::size_t>
class IncrementalStatistics
{
public:
    typedef val    ValueType;
    typedef size_t SizeType;

private:
    SizeType  n     = 0;   // Counter for increment. computation of mean and variance
    ValueType mu    = 0.0; // Mean value of data
    ValueType delta = 0.0; // Distance between mean and current value
    ValueType m2    = 0.0; // Sum of delta squares
public:
    IncrementalStatistics() {}

    template <typename Range>
    explicit IncrementalStatistics(const Range& xs)
        : IncrementalStatistics(xs.begin(), xs.end())
    {
    }

    template <typename Iter>
    IncrementalStatistics(Iter i, Iter j)
    {
        for (auto k = i; k != j; ++k)
            apply(*k);
    }

    SizeType  size() const { return n; }
    ValueType mean() const { return mu; }
    ValueType variance() const { return n <= 1 ? 0 : m2 / (n - 1); }
    ValueType sd() const { return sqrt(variance()); }
    ValueType se() const { return sd() / sqrt(size()); }

    void apply(ValueType x)
    {
        n += 1;
        delta = x - mu;
        mu    = mu + delta / n;
        m2    = m2 + delta * (x - mu);
    }
    void                   operator()(ValueType x) { apply(x); }
    IncrementalStatistics& operator+=(ValueType x)
    {
        apply(x);
        return *this;
    }

    void reset()
    {
        n     = 0;
        mu    = 0.0;
        delta = 0.0;
        m2    = 0.0;
    }
};

template <class val = double, class CMP = std::less<val>>
class IncrementalMinMax
{
public:
    typedef val ValueType;
    typedef CMP ComparatorType;

private:
    ComparatorType cmp;
    ValueType      minimal = std::numeric_limits<ValueType>::max();
    ValueType      maximal = std::numeric_limits<ValueType>::lowest();

public:
    IncrementalMinMax() {}

    template <typename Range>
    explicit IncrementalMinMax(const Range& xs) : IncrementalMinMax(xs.begin(), xs.end())
    {
    }

    template <typename Iter>
    IncrementalMinMax(Iter i, Iter j)
    {
        for (auto k = i; k != j; ++k)
            apply(*k);
    }

    ValueType min() const { return minimal; }
    ValueType max() const { return maximal; }
    ValueType median() const { return (max() + min()) / 2; }

    void apply(ValueType x)
    {
        minimal = cmp(x, minimal) ? x : minimal;
        maximal = cmp(maximal, x) ? x : maximal;
    }

    void               operator()(ValueType x) { apply(x); }
    IncrementalMinMax& operator+=(ValueType x)
    {
        apply(x);
        return *this;
    }

    void reset()
    {
        minimal = std::numeric_limits<ValueType>::max();
        maximal = std::numeric_limits<ValueType>::lowest();
    }
};

template <class T = double, class CMP = std::less<T>>
class IncrementalDescription
{
private:
    using ValueType = T;
    using SizeType  = size_t;
    IncrementalMinMax<T, CMP> mm;
    IncrementalStatistics<T>  stats;
    ValueType                 total = 0;

public:
    IncrementalDescription() {}

    template <typename Range>
    explicit IncrementalDescription(const Range& xs)
        : IncrementalDescription(xs.begin(), xs.end())
    {
    }

    template <typename Iter>
    IncrementalDescription(Iter i, Iter j)
    {
        for (auto k = i; k != j; ++k)
            apply(*k);
    }

    ValueType min() const { return mm.min(); }
    ValueType max() const { return mm.max(); }
    ValueType median() const { return mm.median(); }
    SizeType  size() const { return stats.size(); }
    ValueType mean() const { return stats.mean(); }
    ValueType variance() const { return stats.variance(); }
    ValueType sd() const { return stats.sd(); }
    ValueType se() const { return stats.se(); }
    ValueType sum() const { return total; }

    void apply(ValueType x)
    {
        mm += x;
        stats += x;
        total += x;
    }

    void                    operator()(ValueType x) { apply(x); }
    IncrementalDescription& operator+=(ValueType x)
    {
        apply(x);
        return *this;
    }

    void reset()
    {
        mm.reset();
        stats.reset();
        total = 0;
    }
};
} // namespace sd
