#pragma once

#include <bitcontainer/bit_view.hxx>
#include <containers/small_vector.hxx>
#include <marray/marray.hxx>

#include <array>
#include <vector>

namespace sd
{

template <typename T, size_t length>
using static_bitset = sd::bit_view<std::array<T, length>>;

template <typename C, typename = decltype(std::declval<C>().resize(0, 0))>
struct base_bitset : public sd::bit_view<C>
{
    using container_type = C;
    using base           = sd::bit_view<container_type>;
    using block_type     = typename base::block_type;

    using base::base;

    void insert(size_t i)
    {
        if (this->length() <= i)
        {
            auto blocks_needed = base::blocks_needed_to_store_n_bits(i + 1);
            this->container.resize(blocks_needed, 0);
            this->length_ = i + 1;
        }
        this->set(i);
    }

    void erase(size_t i)
    {
        if (i < this->length())
            base::unset(i);
    }
    void insert(size_t i, bool value)
    {
        if (value)
            insert(i);
        else
            erase(i);
    }

    template <typename S>
    void insert(const sd::bit_view<S>& rhs)
    {
        this->container.resize(std::max(this->container.size(), rhs.container.size()), 0);
        this->length_ = std::max(this->length(), rhs.length());
        size_t n      = std::min(this->container.size(), rhs.container.size());
        for (size_t i = 0; i < n; ++i)
        {
            this->container[i] |= rhs.container[i];
        }
    }

    void insert(slice<const size_t> is)
    {
        if (!is.empty())
        {
            reserve(is.back() + 1);
            for (auto i : is.span())
                insert(i);
        }
    }

    void reserve(size_t i) { resize(i); }

    void   reset() { this->clear(); }
    bool   test(size_t i) const { return is_subset(i, *this); }
    size_t count() const { return base::count(); }

    void resize(size_t n, bool value = false)
    {
        n += 1;
        auto k = base::blocks_needed_to_store_n_bits(n);
        this->container.resize(k, value ? ~block_type(0) : block_type(0));
        this->length_ = n;
    }

    // private:
    // base::set;
};

template <typename S, typename T, typename U>
void intersection(const bit_view<S>& x, const bit_view<T>& y, base_bitset<U>& z)
{
    z.resize(std::max(x.length(), y.length()));
    const auto&  a = x.container;
    const auto&  b = y.container;
    auto&        c = z.container;
    const size_t m = std::min(a.size(), b.size());
    for (size_t i = 0; i < m; ++i)
    {
        c[i] = b[i] & a[i];
    }
    // for (size_t i = m; i < c.size(); ++i)
    // {
    //     c[i] = 0;
    // }
    z.zero_unused_bits();
}

template <typename T, typename Alloc = std::allocator<T>>
struct dynamic_bitset : public sd::base_bitset<std::vector<T, Alloc>>
{
    using container_type = std::vector<T>;
    using base           = sd::base_bitset<container_type>;

    explicit dynamic_bitset(size_t highest_bit, bool value = false)
        : base(container_type(base::blocks_needed_to_store_n_bits(highest_bit + 1), 0),
               highest_bit)
    {
        if (value)
        {
            assert(base::count() == 0);
            base::flip_all();
            assert(base::count() == base::length());
        }
    }
    dynamic_bitset() = default;
};

template <typename T, size_t N>
struct hybrid_bitset : public base_bitset<sd::small_vector<T, N>>
{
    using container_type = sd::small_vector<T, N>;
    using base           = sd::base_bitset<container_type>;

    explicit hybrid_bitset(size_t highest_bit, bool value = false)
        : base(container_type(base::blocks_needed_to_store_n_bits(highest_bit + 1), 0),
               highest_bit)
    {
        if (value)
        {
            assert(base::count() == 0);
            base::flip_all();
            assert(base::count() == base::length());
        }
    }
    hybrid_bitset() = default;
};
} // namespace sd