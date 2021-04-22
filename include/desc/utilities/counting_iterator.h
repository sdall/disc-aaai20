#pragma once
#ifndef sd_counting_iterator_h
#define sd_counting_iterator_h

#include <iterator>
#include <type_traits>

namespace sd
{

template <typename T>
class counting_iterator
{
    static_assert(::std::is_integral<T>::value, "integral type is required for counting");

public:
    typedef T                                    value_type;
    typedef const T*                             pointer;
    typedef T                                    reference;
    typedef ::std::random_access_iterator_tag    iterator_category;
    typedef typename ::std::make_signed<T>::type difference_type;

    counting_iterator() : _counter() {}
    explicit counting_iterator(T init) : _counter(init) {}

    reference operator*() const { return _counter; }
    reference operator[](difference_type n) const { return *(*this + n); }
    counting_iterator& operator++() { return *this += 1; }
    counting_iterator& operator--() { return *this -= 1; }
    counting_iterator operator++(int)
    {
        counting_iterator other(*this);
        ++(*this);
        return other;
    }
    counting_iterator operator--(int)
    {
        counting_iterator other(*this);
        --(*this);
        return other;
    }
    counting_iterator& operator-=(difference_type n) { return *this += -n; }
    counting_iterator& operator+=(difference_type n)
    {
        _counter += n;
        return *this;
    }    
    difference_type operator-(const counting_iterator& other) const
    {
        return _counter - other._counter;
    }
    counting_iterator operator-(difference_type n) const
    {
        return counting_iterator(_counter - n);
    }
    counting_iterator operator+(difference_type n) const
    {
        return counting_iterator(_counter + n);
    }
    friend counting_iterator operator+(difference_type n, const counting_iterator other)
    {
        return other + n;
    }

    bool operator==(const counting_iterator& other) const { return *this - other == 0; }
    bool operator!=(const counting_iterator& other) const { return !(*this == other); }
    bool operator<(const counting_iterator& other) const { return *this - other < 0; }
    bool operator>(const counting_iterator& other) const { return other < *this; }
    bool operator<=(const counting_iterator& other) const { return !(*this > other); }
    bool operator>=(const counting_iterator& other) const { return !(*this < other); }

private:
    T _counter;
};

} // namespace sd

#endif