#pragma once
#include <disc/interfaces/BoostMultiprecision.hxx>

#include <disc/desc/Desc.hxx>
#include <disc/disc/Disc.hxx>

#include <bindings/common/TraitBuilder.hxx>

using namespace sd::disc;

template <typename py_list_t, typename S>
auto& pylist_to_itemset(py_list_t& x, S& out)
{
    out.clear();
    for (auto&& i : x)
    {
        out.insert(size_t(i));
    }
    return out;
}

template <typename S, typename py_list_t>
Dataset<S> create_dataset(py_list_t& xss)
{
    itemset<S> buffer;
    Dataset<S> out;
    out.reserve(xss.size());
    for (auto&& xs : xss)
    {
        out.insert(pylist_to_itemset(xs, buffer));
    }
    return out;
}

template <typename py_list_t, typename Trait>
void create_dataset(py_list_t& x, Component<Trait>& c)
{
    c.data = create_dataset<typename Trait::pattern_type>(x);
}
template <typename py_list_t, typename Trait>
void create_dataset(py_list_t& x, Composition<Trait>& c)
{
    using S = typename Trait::pattern_type;
    c.data  = PartitionedData<S>(create_dataset<S>(x));
}

template <typename py_list_t, typename y_py_list_t, typename Trait>
void create_dataset(py_list_t& x, y_py_list_t& y, Composition<Trait>& c)
{
    using S = typename Trait::pattern_type;
    std::vector<size_t> yy(y.size());
    std::copy(y.begin(), y.end(), yy.begin());
    c.data  = PartitionedData<S>(create_dataset<S>(x), yy);
}

// struct Result
// {
//     std::vector<size_t> labels;
//     std::vector<std::vector<size_t>> assignment;
//     std::vector<double> fr;
//     double f, f0;
// };

template <typename trait_type, typename S>
double desc_impl(Dataset<S> const& dataset, std::vector<size_t>  const& labels, size_t min_support)
{
    using namespace sd::disc;
    
    DiscConfig cfg;
    cfg.min_support      = min_support;
    cfg.use_bic          = true;
    cfg.max_factor_size  = 8;
    cfg.max_factor_width = 10;

    if (labels.size() != 0)
    {
        Composition<trait_type> c;
        c.data = dataset;
        initialize_model(c, cfg);
        auto initial_encoding = c.encoding = encode(c, cfg);
        discover_patterns_generic(c, cfg);
        c.initial_encoding = initial_encoding;
        c.data.revert_order();
        return c.encoding.objective();
    }
    else
    {
        Component<trait_type> c;
        c.data  = PartitionedData<S>(dataset, labels);
        initialize_model(c, cfg);
        c.initial_encoding = initial_encoding;
        discover_patterns_generic(c, cfg);
        return c.encoding.objective();
    }
    return 0;
}

template <typename S>
double desc2(Dataset<S> const& dataset, std::vector<size_t>  const& labels, size_t min_support)
{
    return desc_impl<DefaultTrait>(dataset, labels, min_support);
}

template<typename Container>
struct index_iter {
    Container* x;
    size_t n = 0;

    using value_type = std::decay_t<decltype(std::declval<Container>()[0])>;
    using difference_type = long;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    index_iter& operator++() {n = n + 1; return *this;}
    index_iter operator++(int) {auto retval = *this; ++(*this); return retval;}
    bool operator==(index_iter other) const {return n == other.n;}
    bool operator!=(index_iter other) const {return !(*this == other);}
    auto operator*() {return x->operator[](n);}
};

struct PyListWrapper  {
    PyObject * xs;
    size_t size() { return static_cast<size_t>(PyList_Size(xs)); }
    operator const bool() const { return xs; }
    PyObject * operator[](size_t i) 
    {
        return PyList_GetItem(xs, static_cast<Py_ssize_t> (i));
    }
       
};

struct PySize_tListWrapper : PyListWrapper
{
    size_t operator[](size_t i) 
    {
        return PyLong_AsSize_t(PyListWrapper::operator[](i));
    }
    auto begin() { return index_iter<PySize_tListWrapper>{this, 0}; } 
    auto end() { return index_iter<PySize_tListWrapper>{this, size()}; } 
};

struct PyDataWrapper : PyListWrapper
{
    PySize_tListWrapper operator[](size_t i) 
    {
        return {PyListWrapper::operator[](i) };
    }
    auto begin() { return index_iter<PyDataWrapper>{this, 0}; } 
    auto end() { return index_iter<PyDataWrapper>{this, size()}; } 
};

double desc_(PyObject* dataset, PyObject* labels, int min_support)
{
    return desc_impl<DefaultTrait>(PyDataWrapper{dataset}, PySize_tListWrapper{labels}, min_support);
}

double desc_2(PyObject* dataset, PyObject* labels, int min_support)
{
    return desc_impl<DefaultTrait>(PyDataWrapper{dataset}, PySize_tListWrapper{labels}, min_support);
}
