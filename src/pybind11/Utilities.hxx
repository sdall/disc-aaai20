#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <disc/desc/Composition.hxx>

namespace sd::disc::pyutils
{

namespace py = pybind11;

template <typename In, typename Out>
Out& pylist_to_itemset(const In& in, Out& out)
{
    out.clear();
    for (auto x : in)
    {
        out.insert(py::cast<size_t>(x));
    }
    return out;
}

template <typename S>
Dataset<S> create_dataset(const py::list& x)
{
    sd::disc::itemset<S> buffer;
    Dataset<S>           out;
    out.reserve(x.size());
    for (const auto& xs : x)
    {
        out.insert(pylist_to_itemset(xs, buffer));
    }
    return out;
}
template <typename Trait>
void create_dataset(const py::list& x, Component<Trait>& c)
{
    c.data = create_dataset<typename Trait::pattern_type>(x);
}
template <typename Trait>
void create_dataset(const py::list& x, Composition<Trait>& c)
{
    using S = typename Trait::pattern_type;
    c.data = PartitionedData<S>(create_dataset<S>(x));
}
template <typename Trait>
void create_dataset(const py::list& x, const py::list& y, Composition<Trait>& c)
{
    using S = typename Trait::pattern_type;
    c.data = PartitionedData<S>(create_dataset<S>(x), py::cast<std::vector<size_t>>(y) );
}

// template <typename Data>
// void insert_pylists_into_data(const py::list& in, Data& out)
// {
//     sd::disc::itemset<typename Data::pattern_type> buffer;
//     out.reserve(in.size());
//     for (const auto& xs : in)
//     {
//         out.insert(pylist_to_itemset(xs, buffer));
//     }
// }

// template <typename Data>
// void insert_pylists_into_data(Data& data, const py::list& dataset, const py::list& labels)
// {
//     data.reserve(dataset.size());
//     sd::disc::itemset<typename Data::pattern_type> buffer;

//     if (dataset.size() != labels.size())
//         throw std::domain_error("expect: dataset.size() == labels.size()");

//     for (size_t i = 0; i < dataset.size(); ++i)
//     {
//         data.insert(pylist_to_itemset(dataset[i], buffer), py::cast<size_t>(labels[i]));
//     }
//     data.group_by_label();
// }

template <typename Data>
void insert_data_into_pylist(const Data& in, const py::list& out)
{
    using namespace sd::disc;
    for (const auto& x : in)
    {
        py::list xx;
        sd::foreach(point(x), [&](size_t i) { xx.append(i); });
        out.append(std::move(xx));
    }
}

template <typename Trait>
py::dict translate_to_pydict(Component<Trait> const& co)
{
    py::list patternset;
    insert_data_into_pylist(co.summary, patternset);
    auto            frequencies = py::array_t<double>(co.summary.size());
    py::buffer_info info        = frequencies.request();
    auto target = static_cast<double*>(info.ptr);
    for(size_t i = 0; i < co.summary.size(); ++i) 
    {
        target[i] = static_cast<double>(co.summary.label(i));
    }    

    py::dict r;

    r["pattern_set"]       = patternset;
    r["frequencies"]       = frequencies;
    r["initial_objective"] = co.initial_encoding.objective();
    r["objective"]         = co.encoding.objective();

    return r;
}

template <typename Trait>
py::dict translate_to_pydict(Composition<Trait> const& co)
{
    py::list patternset;
    insert_data_into_pylist(co.summary, patternset);
    auto frequencies = py::array_t<double>();
    frequencies.resize({co.frequency.extent(0), co.frequency.extent(1)});
    auto target = static_cast<double*>(frequencies.request().ptr);
    for(size_t i = 0; i < co.frequency.size(); ++i) 
    {
        target[i] = static_cast<double>(co.frequency.data()[i]);
    }
    py::list assignment_matrix;
    for (const auto& r : co.assignment)
    {
        py::list xx;
        for (auto i : r.container)
        {
            xx.append(i);
        };
        assignment_matrix.append(std::move(xx));
    }

    auto labels = py::array_t<int>(co.data.size());
    {
        py::buffer_info info = labels.request();
        std::copy_n(
            co.data.template col<0>().data(), co.data.size(), static_cast<int*>(info.ptr));
    }

    py::dict r;

    r["pattern_set"]       = patternset;
    r["frequencies"]       = frequencies;
    r["assignment_matrix"] = assignment_matrix;
    r["labels"]            = labels;
    r["initial_objective"] = co.initial_encoding.objective();
    r["objective"]         = co.encoding.objective();

    return r;
}

} // namespace sd::disc::pyutils
