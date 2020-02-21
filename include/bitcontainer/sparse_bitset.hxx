#pragma once

#include <containers/small_vector.hxx>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace sd
{

template <typename InputIt1, typename InputIt2>
auto size_of_intersection(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
{
    std::size_t result = 0;
    while (first1 != last1 && first2 != last2)
    {
        if (*first1 == *first2)
            result++;
        if (*first1 < *first2)
            ++first1;
        else
            ++first2;
    }
    return result;
}

template <typename InputIt1, typename InputIt2>
bool intersects(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
{
    while (first1 != last1 && first2 != last2)
    {
        if (*first1 == *first2)
            return true;
        if (*first1 < *first2)
            ++first1;
        else
            ++first2;
    }
    return false;
}

template <typename Container>
struct sparse_bit_view
{
    using container_type = Container;
    using size_type      = typename container_type::size_type;

    decltype(auto) data() const { return container.data(); }
    size_type      size() const { return container.size(); }
    size_type      count() const { return size(); }
    bool           empty() const { return container.empty(); }

    bool contains(size_type i) const
    {
        return std::binary_search(container.begin(), container.end(), i);
    }

    container_type container;
};

template <typename T>
struct bit_view;

template <typename C>
struct base_bitset_sparse : sparse_bit_view<C>
{
    using container_type = typename sparse_bit_view<C>::container_type;
    using size_type      = typename sparse_bit_view<C>::size_type;
    using sparse_bit_view<C>::sparse_bit_view;
    using sparse_bit_view<C>::container;

    void reserve(size_t n) { container.reserve(n); }

    void insert(size_type i)
    {
        auto pos = std::lower_bound(container.begin(), container.end(), i);
        if (pos == container.end() || *pos != i)
            container.insert(pos, i);
    }
    void erase(size_type i)
    {
        auto pos = std::lower_bound(container.begin(), container.end(), i);
        if (pos != container.end() && *pos == i)
            container.erase(pos);
    }
    void insert(size_type i, bool value)
    {
        if (value)
            insert(i);
        else
            erase(i);
    }

    template <typename T>
    void insert(const sd::sparse_bit_view<T>& rhs)
    {
        auto&  a = this->container;
        size_t n = a.size();
        a.reserve(n + rhs.size());
        a.insert(a.end(), rhs.container.begin(), rhs.container.end());
        std::inplace_merge(a.begin(), a.begin() + n, a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        // assert(std::is_sorted(container.begin(), container.end()));
    }

    template <typename T>
    void insert(const sd::bit_view<T>& rhs)
    {
        iterate_over(rhs, [&](size_t i) { insert(i); });
    }

    void insert(const cpslice<size_t> rhs)
    {
        auto&  a = this->container;
        size_t n = a.size();
        a.reserve(n + rhs.size());
        a.insert(a.end(), rhs.begin(), rhs.end());
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        // assert(std::is_sorted(container.begin(), container.end()));
    }
    void clear() { container.clear(); }

    auto data() const { return container.data(); }

    decltype(auto) operator[](size_t i) const { return container[i]; }

    decltype(auto) begin() const { return container.begin(); }
    decltype(auto) end() const { return container.end(); }
};

template <typename S>
std::size_t last_entry(const sparse_bit_view<S>& s)
{
    assert(!s.empty());
    return s.container.back();
}

template <typename S, typename T>
bool is_subset(sparse_bit_view<S> const& s, sparse_bit_view<T> const& t)
{
    if (s.empty())
        return true;
    if (s.size() == 1)
    {
        return t.contains(s.container.front());
    }
    // assert(std::is_sorted(t.container.begin(), t.container.end()));
    // assert(std::is_sorted(s.container.begin(), s.container.end()));
    // assert(std::adjacent_find(s.container.begin(), s.container.end()) == s.container.end());
    // assert(std::adjacent_find(t.container.begin(), t.container.end()) == t.container.end());
    return std::includes(
        t.container.begin(), t.container.end(), s.container.begin(), s.container.end());
}

template <typename S, typename T>
bool is_proper_subset(sparse_bit_view<S> const& s, sparse_bit_view<T> const& t)
{
    return s.size() < t.size() && is_subset(s, t);
}

template <typename S>
bool is_subset(size_t i, sparse_bit_view<S> const& s)
{
    return s.contains(i);
}

template <typename S, typename T>
bool intersects(sparse_bit_view<S> const& s, sparse_bit_view<T> const& t)
{
    return intersects(
        s.container.begin(), s.container.end(), t.container.begin(), t.container.end());
}

template <typename S, typename T, typename U>
void intersection(const base_bitset_sparse<S>& x,
                  const base_bitset_sparse<T>& y,
                  base_bitset_sparse<U>&       intersection)
{
    intersection.clear();
    intersection.reserve(std::min(x.size(), y.size()));
    std::set_intersection(x.container.begin(),
                          x.container.end(),
                          y.container.begin(),
                          y.container.end(),
                          std::back_inserter(intersection.container));
}

template <typename S, typename T>
void intersection(const sparse_bit_view<S>& s, sparse_bit_view<T>& t)
{
    T next;
    // assert(std::is_sorted(s.container.begin(), s.container.end()));
    // assert(std::is_sorted(t.container.begin(), t.container.end()));
    next.reserve(t.size());
    std::set_intersection(t.container.begin(),
                          t.container.end(),
                          s.container.begin(),
                          s.container.end(),
                          std::back_inserter(next));
    t.container = std::move(next);
}

template <typename S, typename T>
auto size_of_intersection(const sparse_bit_view<S>& s, const sparse_bit_view<T>& t)
{
    return size_of_intersection(
        s.container.begin(), s.container.end(), t.container.begin(), t.container.end());
}

template <typename S, typename T>
auto similarity(const sparse_bit_view<S>& s, const sparse_bit_view<T>& t)
{
    return size_of_intersection(s, t);
}

template <typename S, typename T>
bool equal(const sparse_bit_view<S>& s, const sparse_bit_view<T>& t)
{
    return std::equal(
        s.container.begin(), s.container.end(), t.container.begin(), t.container.end());
}

template <typename S, typename Fn>
void iterate_over(const sparse_bit_view<S>& s, Fn&& fn)
{
    for (const auto& i : s.container)
    {
        fn(i);
    }
}

// t <- t \ s
template <typename S, typename T>
void setminus(sparse_bit_view<S>& s, const sparse_bit_view<T>& t)
{
    T next;
    next.reserve(s.size());
    std::set_difference(s.container.begin(),
                        s.container.end(),
                        t.container.begin(),
                        t.container.end(),
                        std::back_inserter(next));
    s.container = std::move(next);
}

// u <- s \ t
template <typename S, typename T, typename U>
void setminus(const sparse_bit_view<S>& s, const sparse_bit_view<T>& t, sparse_bit_view<U>& u)
{
    u.container.clear();
    u.container.reserve(t.size());
    std::set_difference(s.container.begin(),
                        s.container.end(),
                        t.container.begin(),
                        t.container.end(),
                        std::back_inserter(u.container));
}

template <typename S>
size_t count(const sparse_bit_view<S>& s)
{
    return s.size();
}

template <typename S>
bool is_singleton(const sparse_bit_view<S>& s)
{
    return count(s) == 1;
}

template <typename S>
size_t front(const sparse_bit_view<S>& s)
{
    return s.container[0];
}

template <typename T, size_t N>
using sparse_bitset = sd::base_bitset_sparse<std::array<T, N>>;

template <typename T, typename Alloc = std::allocator<T>>
struct sparse_dynamic_bitset : base_bitset_sparse<std::vector<T, Alloc>>
{
    using container_type = std::vector<T, Alloc>;
    using base           = base_bitset_sparse<std::vector<T, Alloc>>;

    explicit sparse_dynamic_bitset(size_t num_bits) { base::container.reserve(num_bits + 1); }
    sparse_dynamic_bitset() = default;
};

template <typename T, size_t N>
struct sparse_hybrid_bitset : public base_bitset_sparse<sd::small_vector<T, N>>
{
    using container_type = sd::small_vector<T, N>;
    using base           = base_bitset_sparse<sd::small_vector<T, N>>;

    explicit sparse_hybrid_bitset(size_t num_bits) { base::container.reserve(num_bits + 1); }
    template <typename Iter>
    sparse_hybrid_bitset(Iter f, Iter l)
    {
        base::container.append(f, l);
        std::sort(this->begin(), this->end());
    }
    sparse_hybrid_bitset() = default;
};

} // namespace sd