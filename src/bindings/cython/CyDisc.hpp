#pragma once
#include <desc/utilities/BoostMultiprecision.hxx>

#include <desc/Desc.hxx>
#include <disc/Disc.hxx>
#include <pybind11/pybind11.h>

#include <bindings/common/TraitBuilder.hxx>
#include <bindings/python/Utilities.hxx>

namespace py = pybind11;
using namespace sd::disc::pyutils;

namespace sd::disc
{

template <typename S>
auto desc(Dataset<S>&       dataset,
          std::vector<int>& labels,
          size_t            min_support,
          bool              use_higher_precision_floats = false)
{
    py::dict r;
    sd::disc::select_real_type<S>(use_higher_precision_floats, [&](auto trait) {
        std::vector<size_t> yy(labels.begin(), labels.end());
        r = pyutils::desc_impl<decltype(trait)>(dataset, yy, min_support);
    });

    return PyDict_Copy(r.ptr());
}

template <typename S>
auto disc(Dataset<S> x, size_t min_supp, double alpha, bool use_higher_precision_floats = false)
{
    py::dict r;
    sd::disc::select_real_type<S>(use_higher_precision_floats, [&](auto trait) {
        r = pyutils::disc_impl<decltype(trait)>(std::move(x), min_supp, alpha);
    });
    return PyDict_Copy(r.ptr());
}

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
    c.data = PartitionedData<S>(create_dataset<S>(x), yy);
}

template <typename Container>
struct index_iter
{
    Container* x;
    size_t     n = 0;

    using value_type        = std::decay_t<decltype(std::declval<Container>()[0])>;
    using difference_type   = long;
    using pointer           = const value_type*;
    using reference         = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    index_iter& operator++()
    {
        n = n + 1;
        return *this;
    }
    index_iter operator++(int)
    {
        auto retval = *this;
        ++(*this);
        return retval;
    }
    bool operator==(index_iter other) const { return n == other.n; }
    bool operator!=(index_iter other) const { return !(*this == other); }
    auto operator*() { return x->operator[](n); }
};

struct PyListWrapper
{
    PyObject* xs;
    size_t    size() { return static_cast<size_t>(PyList_Size(xs)); }
              operator const bool() const { return xs; }
    PyObject* operator[](size_t i) { return PyList_GetItem(xs, static_cast<Py_ssize_t>(i)); }
};

struct PySize_tListWrapper : PyListWrapper
{
    size_t operator[](size_t i) { return PyLong_AsSize_t(PyListWrapper::operator[](i)); }
    auto   begin() { return index_iter<PySize_tListWrapper>{this, 0}; }
    auto   end() { return index_iter<PySize_tListWrapper>{this, size()}; }
};

struct PyDataWrapper : PyListWrapper
{
    PySize_tListWrapper operator[](size_t i) { return {PyListWrapper::operator[](i)}; }
    auto                begin() { return index_iter<PyDataWrapper>{this, 0}; }
    auto                end() { return index_iter<PyDataWrapper>{this, size()}; }
};

// auto desc3(PyObject* dataset, PyObject* labels, int min_support, bool is_sparse, bool use_higher_precision_float)
// {
//     // auto x = py::reinterpret_borrow<py::object>(dataset);
//     // auto y = py::reinterpret_borrow<py::object>(labels);

//     py::object x = py::cast(dataset);
//     py::object y = py::cast(dataset);
//     auto dict = pyutils::(x, y, min_support, is_sparse, use_higher_precision_float);
//     return PyDict_Copy(dict.ptr());
// }

// auto disc3(PyObject* dataset, int min_support, bool is_sparse, bool use_higher_precision_float)
// {
//     // py::object x(dataset, true);
//     auto dict = pyutils::discover_composition(py::cast(dataset), min_support, is_sparse, use_higher_precision_float);
//     return PyDict_Copy(dict.ptr());
// }

auto desc2(PyObject* dataset, PyObject* labels, int min_support)
{
    auto xx = PyDataWrapper{dataset};
    auto yy = PySize_tListWrapper{labels};

    auto x = create_dataset<typename DefaultTrait::pattern_type>(xx);
    auto y = std::vector<size_t>(yy.begin(), yy.end());

    return PyDict_Copy(desc_impl<DefaultTrait>(x, y, min_support).ptr());
}

auto cy_desc_from_matrix(PyObject* dataset, PyObject* labels, int min_support, bool is_sparse, bool use_higher_precision_float)
{
    py::dict r;

    auto dataset_11 = py::reinterpret_borrow<py::array_t<bool>>(dataset);
    auto labels_11 = py::reinterpret_borrow<py::list>(labels);

    std::vector<size_t> y;

    if(labels && labels_11 && labels_11.size() > 0)
    {
        y = py::cast<std::vector<size_t>>(labels_11);
    }

    if(!dataset || !dataset_11 ||  dataset_11.size() == 0)
    {
        return PyDict_Copy(r.ptr());
    }

    sd::disc::build_trait(is_sparse, use_higher_precision_float, [&](auto trait) {
        using T = decltype(trait);
        using S = typename T::pattern_type;
        auto x  = pyutils::create_dataset<S>(dataset_11);
        r       = pyutils::desc_impl<T>(std::move(x), y, min_support);
    });
    return PyDict_Copy(r.ptr());
}

auto cy_desc_from_transactions(PyObject* dataset, PyObject* labels, int min_support, bool is_sparse, bool use_higher_precision_float)
{
    py::dict r;

    auto dataset_11 = py::reinterpret_borrow<py::list>(dataset);
    auto labels_11 = py::reinterpret_borrow<py::list>(labels);

    std::vector<size_t> y;

    if(labels && labels_11 && labels_11.size() > 0)
    {
        y = py::cast<std::vector<size_t>>(labels_11);
    }

    if(!dataset || !dataset_11 || dataset_11.size() == 0)
    {
        return PyDict_Copy(r.ptr());
    }

    sd::disc::build_trait(is_sparse, use_higher_precision_float, [&](auto trait) {
        using T = decltype(trait);
        using S = typename T::pattern_type;
        auto x  = pyutils::create_dataset<S>(dataset_11);
        r       = pyutils::desc_impl<T>(std::move(x), y, min_support);
    });
    return PyDict_Copy(r.ptr());
}

auto cy_disc_from_matrix(PyObject* dataset, int min_support, bool is_sparse, bool use_higher_precision_float)
{
    py::dict r;

    auto dataset_11 = py::reinterpret_borrow<py::array_t<bool>>(dataset);

    if(!dataset || !dataset_11 ||  dataset_11.size() == 0)
    {
        return PyDict_Copy(r.ptr());
    }

    sd::disc::build_trait(is_sparse, use_higher_precision_float, [&](auto trait) {
        using T = decltype(trait);
        using S = typename T::pattern_type;
        auto x  = pyutils::create_dataset<S>(dataset_11);
        r       = pyutils::disc_impl<T>(std::move(x), min_support, 0.05);
    });
    return PyDict_Copy(r.ptr());
}

auto cy_disc_from_transactions(PyObject* dataset, PyObject* labels, int min_support, bool is_sparse, bool use_higher_precision_float)
{
    py::dict r;

    auto dataset_11 = py::reinterpret_borrow<py::list>(dataset);

    if(!dataset || !dataset_11 || dataset_11.size() == 0)
    {
        return PyDict_Copy(r.ptr());
    }

    sd::disc::build_trait(is_sparse, use_higher_precision_float, [&](auto trait) {
        using T = decltype(trait);
        using S = typename T::pattern_type;
        auto x  = pyutils::create_dataset<S>(dataset_11);
        r       = pyutils::disc_impl<T>(std::move(x), min_support, 0.05);
    });
    return PyDict_Copy(r.ptr());
}



// auto disc3(PyObject* dataset, int min_support, bool is_sparse, bool use_higher_precision_float)
// {
//     // py::object x(dataset, true);
//     auto dict = pyutils::discover_composition(py::cast(dataset), min_support, is_sparse, use_higher_precision_float);
//     return PyDict_Copy(dict.ptr());
// }




} // namespace sd::disc