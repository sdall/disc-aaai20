#pragma once

#include <bitcontainer/bitset.hxx>
#include <bitcontainer/extra/fwd_iterator.hxx>
#include <bitcontainer/sparse_bitset.hxx>

#include <numeric>
#include <type_traits>
#include <vector>

namespace sd
{
namespace disc
{

struct tag_sparse
{
};
struct tag_dense
{
};

template <typename Tag>
constexpr bool is_sparse(Tag&&)
{
    return std::is_same<Tag, tag_sparse>();
}

template <typename Tag>
constexpr bool is_dense(Tag&&)
{
    return std::is_same<Tag, tag_dense>();
}

using index_type = std::size_t;

#if defined(USE_LONG_INDEX)
using sparse_index_type = std::size_t;
#else
using sparse_index_type = std::uint16_t;
#endif

template <typename T>
using storage_container = std::conditional_t<is_sparse(T{}),
                                             sparse_dynamic_bitset<sparse_index_type>,
                                             hybrid_bitset<std::uint64_t, 4>>;
template <typename T>
using itemset = std::conditional_t<is_sparse(T{}),
                                   sparse_dynamic_bitset<sparse_index_type>,
                                   hybrid_bitset<std::uint64_t, 4>>;
template <typename T>
using long_storage_container = std::conditional_t<is_sparse(T{}),
                                                  sparse_dynamic_bitset<std::uint32_t>,
                                                  hybrid_bitset<std::uint64_t, 8>>;
} // namespace disc

template <typename S>
std::size_t get_dim(const bit_view<S>& s)
{
    return s.empty() ? 0 : (last_entry(s) + 1);
}

template <typename S>
std::size_t get_dim(const sparse_bit_view<S>& s)
{
    return !s.container.empty() ? (s.container.back() + 1) : 0;
}

} // namespace sd
