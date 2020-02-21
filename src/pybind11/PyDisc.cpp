#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <disc/disc/Disc.hxx>
#include <disc/disc/Desc.hxx>

namespace py = pybind11;

using trait_type = sd::disc::DefaultTrait;
using pattern_type = typename trait_type::pattern_type;
using float_type = typename trait_type::float_type;
// using trait_type = sd::disc::Trait<pattern_type, float_type, sd::disc::MEDistribution<pattern_type, float_type>>;

auto discover_patternset(const py::list& dataset, double alpha, size_t min_support) {
    using namespace sd::disc;

    PatternsetResult<trait_type> co;
    co.data.reserve(dataset.size());
    itemset<pattern_type> buffer;
    for(const auto& xs : dataset) 
    {
        buffer.clear();
        for(auto x : xs) {buffer.insert(py::cast<size_t>(x)); }
        co.data.insert(buffer);
    }

    MiningSettings cfg;
    cfg.alpha = alpha;
    cfg.min_support = min_support;
    cfg.use_bic = true;

    co = sd::disc::discover_patternset(std::move(co), cfg);

    py::list patternset;
    for(const auto &x : co.summary.col<1>()) {
        py::list xx;
        iterate_over(x, [&](size_t i) { xx.append(i); });
        patternset.append(std::move(xx));
    }
    
    auto frequencies = py::array_t<float_type>(co.summary.size());
    py::buffer_info info = frequencies.request();
    std::copy_n(co.summary.col<0>().data(), co.summary.size(), static_cast<float_type*>(info.ptr));

    return py::make_tuple(patternset, 
                          frequencies, 
                          co.initial_encoding.objective(), 
                          co.encoding.objective());
}

auto characterize_partitions(const py::list& dataset, const py::list& labels, double alpha, size_t min_support) {
    using namespace sd::disc;

    Composition<trait_type> co;
    co.data.reserve(dataset.size());
    itemset<pattern_type> buffer;
    
    if(dataset.size() != labels.size())
        throw std::domain_error("expect: dataset.size() == labels.size()");
    
    for(size_t i = 0; i < dataset.size(); ++i) 
    {
        const auto& xs = dataset[i];

        buffer.clear();
        for(auto x : xs) {buffer.insert(py::cast<size_t>(x)); }

        co.data.insert(buffer, py::cast<size_t>(labels[i]));
        // co.data.insert(buffer);
    }


    MiningSettings cfg;
    cfg.alpha = alpha;
    cfg.min_support = min_support;
    cfg.use_bic = true;

    // initialize_composition(co, cfg);
    // co = discover_patternsets(std::move(co));

    // for(size_t i = 0; i < dataset.size(); ++i) {
    //     co.data.label(i) = py::cast<size_t>(labels[i]);
    // }

    initialize_composition(co, cfg);
    auto initial_encoding = co.encoding;
    co = discover_patternsets(std::move(co), cfg);

    py::list patternset;
    for(const auto &x : co.summary.col<1>()) {
        py::list xx;
        iterate_over(x, [&](size_t i) { xx.append(i); });
        patternset.append(std::move(xx));
    }
    
    auto frequencies = py::array_t<float_type>();
    frequencies.resize({co.frequency.extent(0), co.frequency.extent(1)});
    py::buffer_info info = frequencies.request();
    std::copy_n(co.frequency.data(), co.frequency.size(), static_cast<float_type*>(info.ptr));

    py::list assignment_matrix;
    
    for(const auto& r : co.assignment) {
        py::list row;
        for (auto i : r.container) { row.append(py::cast<int>(i));}
        assignment_matrix.append(std::move(row));
    }

    return py::make_tuple(patternset, 
                          frequencies, 
                          assignment_matrix,
                          initial_encoding.objective(), 
                          co.encoding.objective());
}

auto discover_composition(const py::list& dataset, double alpha, size_t min_support) {
    using namespace sd::disc;

    Composition<trait_type> co;
    co.data.reserve(dataset.size());
    itemset<pattern_type> buffer;
    
    for(const auto& xs : dataset) 
    {
        buffer.clear();
        for(auto x : xs) {buffer.insert(py::cast<size_t>(x)); }
        co.data.insert(buffer);
    }

    DecompositionSettings cfg;
    cfg.alpha = alpha;
    cfg.min_support = min_support;
    cfg.use_bic = true;

    initialize_composition(co, cfg);
    auto initial_encoding = co.encoding;
    co = mine_split_round_repeat(std::move(co), cfg);

    py::list patternset;
    for(const auto &x : co.summary.col<1>()) {
        py::list xx;
        iterate_over(x, [&](size_t i) { xx.append(i); });
        patternset.append(std::move(xx));
    }
    
    auto frequencies = py::array_t<float_type>();
    {
        frequencies.resize({co.frequency.extent(0), co.frequency.extent(1)});
        py::buffer_info info = frequencies.request();
        std::copy_n(co.frequency.data(), co.frequency.size(), static_cast<float_type*>(info.ptr));
    }
    
    py::list assignment_matrix;
    
    for(const auto& r : co.assignment) {
        py::list row;
        for (auto i : r.container) { row.append(py::cast<int>(i));}
        assignment_matrix.append(std::move(row));
    }

    auto labels = py::array_t<int>(co.data.size());
    {
        py::buffer_info info = labels.request();
        std::copy_n(co.data.col<0>().data(), co.data.size(), static_cast<int*>(info.ptr));
    }

    return py::make_tuple(patternset, 
                          frequencies, 
                          assignment_matrix,
                          labels,
                          initial_encoding.objective(), 
                          co.encoding.objective());
}

struct PyMEDistribution : sd::disc::MEDistribution<pattern_type, float_type> {
    using base = sd::disc::MEDistribution<pattern_type, float_type>;
    using base::float_type;

    PyMEDistribution(size_t dim) : base(dim, 0) {
        sd::disc::estimate_model(*this);
    }

    void insert(float_type label, const py::list& t)
    {
        buf.clear();
        for(auto i : t) {
            buf.insert(py::cast<size_t>(i));
        }

        base::insert(label, buf);
        sd::disc::estimate_model(*this);
    }

    void insert_batch(py::array_t<double>& ys, const py::list& ts)
    {
        py::buffer_info nfo = ys.request();
        auto y = sd::slice<const double>(static_cast<const double*>(nfo.ptr), ys.size());
        for(size_t i = 0; i < ts.size(); ++i) {
            buf.clear();
            for(auto j : ts[i]) {
                buf.insert(py::cast<size_t>(j));
            }
            base::insert(y[i], buf);
        }
        sd::disc::estimate_model(*this);
    }

    // void insert_batch(py::list ys, const py::list& ts)
    // {
    //     for(size_t i = 0; i < ts.size(); ++i) {
    //         buf.clear();
    //         for(auto j : ts[i]) {
    //             buf.insert(py::cast<size_t>(j));
    //         }
    //         base::insert(py::cast<double>(ys[i]), buf);
    //     }
    //     sd::disc::estimate_model(*this);
    // }

    float_type infer_generalized_itemset(const py::list& t) const
    {
        thread_local sd::disc::itemset<pattern_type> buf;
        buf.clear();
        for(auto i : t) {
            buf.insert(py::cast<size_t>(i));
        }
        return base::expected_generalized_frequency(buf);
    }

    float_type infer(const py::list& t) const
    {
        thread_local sd::disc::itemset<pattern_type> buf;
        buf.clear();
        for(auto i : t) {
            buf.insert(py::cast<size_t>(i));
        }
        return base::expected_frequency(buf);
    }

private:
    sd::disc::itemset<pattern_type> buf;
};

PYBIND11_MODULE(disc, m) {
    using namespace pybind11::literals;

    m.doc() = R"doc(    
              DISC: Patternsets and Pattern-Composition
              -----------------------------------------
              This module contains the python interface to DISC, a method for discovering the pattern composition of a dataset.
              The pattern composition consists the following:
                  (1) an interpretable partitioning of the data into components in which patterns follow a significantly different distribution
                  (2) an description of the partitioning using characteristic and shared patterns
            )doc";

    m.def("discover_patternset", &::discover_patternset, "A function that discovers significant patterns for a dataset using the maximum entropy distribution",
          "dataset"_a, "alpha"_a=0.05, "min_support"_a=1);
    m.def("characterize_partitions", &characterize_partitions, "A function that discovers significant patterns and characterizes multiple, given a partition using the maximum entropy distribution",
          "dataset"_a, "partition_label"_a, "alpha"_a=0.05, "min_support"_a=1);
    m.def("discover_composition", &discover_composition, "A function that discovers differently distributed partitions of the dataset as well as significant patterns and characterizes the partitions using the maximum entropy distribution",
          "dataset"_a, "alpha"_a=0.05, "min_support"_a=1);

    py::class_<PyMEDistribution>(m, "Distribution")
        .def(py::init<size_t>())
        .def("infer", &PyMEDistribution::infer)
        .def("infer_generalized_itemset", &PyMEDistribution::infer_generalized_itemset)
        .def("insert_batch", &PyMEDistribution::insert_batch)
        .def("insert", &PyMEDistribution::insert);

    m.attr("__version__") = "dev";
}

