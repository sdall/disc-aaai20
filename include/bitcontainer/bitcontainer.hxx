#pragma once

#include "bitset.hxx"
#include "sparse_bitset.hxx"

namespace sd
{

// template <typename Container,
//           typename S,
//           typename = decltype(std::declval<Container>().begin()),
//           typename = decltype(std::declval<Container>().back()),
//           typename = decltype(std::declval<S>().resize(size_t{}, 0))>
// void copy_from_index_list(const Container& x, base_bitset<S>& output)
// {
//     // assert(std::is_sorted(x.begin(), x.end()));
//     // output.clear();
//     output.resize(x.back() + 1, 0);
//     for (const auto& i : x) { output.insert(i); }
// }

// template <typename S, typename T, typename = decltype(std::declval<S>().reserve(0))>
// void insert(sd::bit_view<S>& into, const sd::sparse_bit_view<S>& rhs)
// {
//     copy_from_index_list(rhs.container, into);
// }

// template <typename S, typename T>
// void insert(sd::bit_view<S>& into, const cpslice<T>& rhs)
// {
//     copy_from_index_list(rhs, into);
// }

// template <typename S, typename T, typename = decltype(std::declval<S>().reserve(0))>
// void insert(sd::base_bitset_sparse<S>& into, const bit_view<T>& rhs)
// {
//     auto&  a = into.container;
//     size_t n = a.size();
//     a.reserve(n + count(rhs));
//     iterate_over(rhs, [&](size_t i) { into.insert(i); });
// }

// template <typename T, typename S>
// void convert_insert_into(bit_view<T> const& b, base_bitset<S> & into)
// {
//     into.insert(b);
// }
// template <typename T, typename S>
// void convert_insert_into(sparse_bit_view<T> const& b, base_bitset<S> & into)
// {
//     into.insert(b.container);
//     // copy_from_index_list(rhs.container, into);
// }
// template <typename S>
// void convert_insert_into(const std::vector<std::size_t>& is, base_bitset<S> & into)
// {
//     into.insert(is);
//     // for (auto i : is) { into.insert(i); }
// }
// template <typename S>
// void convert_insert_into(std::size_t i, base_bitset<S> & into) { into.insert(i); }

// template <typename T, typename S>
// void convert_insert_into(bit_view<T> const& b, base_bitset_sparse<S> & into)
// {
//     // sd::insert(into, b);
//     into.insert(b);
// }
// template <typename T, typename S>
// void convert_insert_into(sparse_bit_view<T> const& b, base_bitset_sparse<S> & into)
// {
//     into.insert(b);
// }
// template <typename S>
// void convert_insert_into(const std::vector<std::size_t>& is, base_bitset_sparse<S> & into)
// {
//     into.insert(is);
// }
// template <typename S>
// void convert_insert_into(std::size_t i, base_bitset_sparse<S> & into) { into.insert(i); }

// template <typename T, bool sparse>
// using get_observer_type = std::conditional_t<sparse,
//                                              sparse_bit_view<cpslice<const T>>,
//                                              bit_view<cpslice<const T>>>;

// template <typename T, bool sparse>
// using get_storage_type =
//     std::conditional_t<sparse, sparse_dynamic_bitset<T>, dynamic_bitset<T>>;

// template <typename T, bool sparse, size_t N>
// using get_hybrid_storage_type =
//     std::conditional_t<sparse, sparse_hybrid_bitset<T, N>, hybrid_bitset<T, N>>;

// template <typename T = tag_dense>
// using dynamic_set = get_storage_type<size_t, is_sparse(T{})>;

// template <typename T = tag_dense, size_t N = 5>
// using small_set = get_hybrid_storage_type<size_t, is_sparse(T{}), N>;

// template <typename T = size_t, bool sparse = false, size_t N = 5>
// struct set_buffer_impl : public get_hybrid_storage_type<T, sparse, N>
// {
//     using base = get_hybrid_storage_type<T, sparse, N>;
//     using base::base;

//     template <typename S>
//     void insert(S&& s)
//     {
//         insert(*static_cast<base*>(this), std::forward<S>(s));
//     }
// };

// template <typename T = tag_dense>
// using set_buffer = set_buffer_impl<size_t, is_sparse(T{}), 5>;

// template <typename S, typename... T>
// base_bitset_sparse<S>& assign(base_bitset_sparse<S>& t, T&&... is)
// {
//     t.clear();
//     [[maybe_unused]] bool b[] = {true, (sd::insert(t, std::forward<T>(is)), true)...};
//     return t;
// }

// template <typename S, typename... T>
// bit_view<S>& assign(bit_view<S>& t, T&&... is)
// {
//     t.clear();
//     [[maybe_unused]] bool b[] = {true, (sd::insert(t, std::forward<T>(is)), true)...};
//     return t;
// }

// template <typename S, size_t N>
// base_bitset_sparse<S>& assign(base_bitset_sparse<S>& t,
//                                             std::array<size_t, N> const&         is)
// {
//     t.clear();
//     for (auto i : is) t.insert(i);
//     return t;
// }

// template <typename S, size_t N>
// bit_view<S>& assign(bit_view<S>& t, std::array<size_t, N> const& is)
// {
//     t.clear();
//     for (auto i : is) t.insert(i);
//     return t;
// }

} // namespace sd