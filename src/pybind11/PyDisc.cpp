
#include <disc/interfaces/BoostMultiprecision.hxx>

#include <disc/desc/Desc.hxx>
#include <disc/disc/Disc.hxx>
#include <utils/TraitBuilder.hxx>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Utilities.hxx"

namespace py = pybind11;

using namespace sd::disc::pyutils;

template <typename trait_type>
auto discover_patternset_(const py::list& dataset, size_t min_support)
{
    using namespace sd::disc;

    DiscConfig cfg;
    cfg.min_support = min_support;
    cfg.use_bic     = true;

    Component<trait_type> co;
    create_dataset(dataset, co);

    discover_patterns_generic(co, cfg);

    return translate_to_pydict(co);
}

auto discover_patternset(const py::list& dataset,
                         size_t          min_support,
                         bool            is_sparse  = false,
                         bool            is_precise = false)
{
    py::dict r;
    sd::disc::build_trait(is_sparse, is_precise, [&](auto trait) {
        r = discover_patternset_<decltype(trait)>(dataset, min_support);
    });
    return r;
}

template <typename trait_type>
auto characterize_partitions_(const py::list& dataset,
                              const py::list& labels,
                              size_t          min_support)
{
    using namespace sd::disc;

    DiscConfig cfg;
    cfg.min_support = min_support;
    cfg.use_bic     = true;

    Composition<trait_type> co;
    create_dataset(dataset, labels, co);
    initialize_model(co, cfg);
    auto initial_encoding = co.encoding = encode(co, cfg);

    discover_patterns_generic(co, cfg);
    co.initial_encoding = initial_encoding;

    return translate_to_pydict(co);
}

auto characterize_partitions(const py::list& dataset,
                             const py::list& labels,
                             size_t          min_support,
                             bool            is_sparse  = false,
                             bool            is_precise = false)
{
    py::dict r;
    sd::disc::build_trait(is_sparse, is_precise, [&](auto trait) {
        r = characterize_partitions_<decltype(trait)>(dataset, labels, min_support);
    });
    return r;
}

template <typename trait_type>
auto discover_composition_(const py::list& dataset, size_t min_support, double alpha)
{
    using namespace sd::disc;

    DiscConfig cfg;
    cfg.alpha            = alpha;
    cfg.min_support      = min_support;
    cfg.use_bic          = true;
    // cfg.max_factor_size  = 5;
    // cfg.max_factor_width = 10;

    Composition<trait_type> co;
    create_dataset(dataset, co);

    initialize_model(co, cfg);
    auto initial_encoding = co.encoding = encode(co, cfg);

    auto pm = [](auto& c, const auto& g) { discover_patterns_generic(c, g); };
    discover_components(co, cfg, pm, sd::EmptyCallback{});
    co.initial_encoding = initial_encoding;

    return translate_to_pydict(co);
}

auto discover_composition(const py::list& dataset,
                          size_t          min_support,
                          double          alpha,
                          bool            is_sparse  = false,
                          bool            is_precise = false)
{
    py::dict r;
    sd::disc::build_trait(is_sparse, is_precise, [&](auto trait) {
        r = discover_composition_<decltype(trait)>(dataset, min_support, alpha);
    });
    return r;
}

template <typename S, typename T>
struct PyMEDist
{
    using float_type   = T;
    using pattern_type = S;

    PyMEDist(size_t dim, size_t max_factor_size = 8, size_t max_factor_width = 12)
        : dist(dim, 1, max_factor_size, max_factor_width)
    {
        sd::disc::estimate_model(dist);
    }

    void insert(float_type label, const py::list& t)
    {
        dist.insert(label, pylist_to_itemset(t, buf), true);
    }

    // void insert_batch(py::array_t<double>& ys, const py::list& ts)
    // {
    //     if (ys.size() != ts.size())
    //     {
    //         throw std::runtime_error("arguments of 'insert_batch' are of different length");
    //     }
    //     py::buffer_info nfo = ys.request();
    //     auto y = sd::slice<const double>(static_cast<const double*>(nfo.ptr), ys.size());
    //     for (size_t i = 0; i < ts.size(); ++i)
    //     {
    //         dist.insert(y[i], pylist_to_itemset(ts[i], buf), true);
    //     }
    // }

    float_type infer_generalized_itemset(const py::list& t) const
    {
        thread_local sd::disc::itemset<S> buf;
        return dist.expectation_generalized_set(pylist_to_itemset(t, buf));
    }

    float_type infer(const py::list& t) const
    {
        thread_local sd::disc::itemset<S> buf;
        return dist.expectation(pylist_to_itemset(t, buf));
    }

    static std::string type_name()
    {
        std::string name = "MEDist_";
        name += sd::disc::storage_type_to_str<S>();
        name += "_";
        name += sd::disc::float_storage_type_to_str<T>();
        return name;
    }

private:
    sd::disc::MaxEntDistribution<S, T> dist;
    sd::disc::itemset<S>               buf;
};

template <typename Trait>
void declare_py_distribution(py::module& m)
{
    using Class = PyMEDist<typename Trait::pattern_type, typename Trait::float_type>;

    py::class_<Class>(m, Class::type_name().c_str())
        .def(py::init<size_t>())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, size_t>())
        .def("infer", &Class::infer)
        .def("infer_generalized_itemset", &Class::infer_generalized_itemset)
        // .def("insert_batch", &Class::insert_batch)
        .def("insert", &Class::insert);
}

PYBIND11_MODULE(disc, m)
{
    using namespace pybind11::literals;

    m.doc() = R"doc(    
              DISC: Patternsets and Pattern-Composition
              -----------------------------------------
              This module contains the python interface to DISC, a method for discovering the pattern composition of a dataset.
              The pattern composition consists the following:
                  (1) an interpretable partitioning of the data into components in which patterns follow a significantly different distribution
                  (2) an description of the partitioning using characteristic and shared patterns
            )doc";

    m.def("discover_patternset",
          &::discover_patternset,
          "A function that discovers significant patterns for a dataset using the maximum "
          "entropy distribution",
          "dataset"_a,
          "min_support"_a = 2,
          "is_sparse"_a   = false,
          "is_precise"_a  = false);
    m.def("characterize_partitions",
          &characterize_partitions,
          "A function that discovers significant patterns and characterizes multiple, given a "
          "partition using the maximum entropy distribution",
          "dataset"_a,
          "partition_label"_a,
          "min_support"_a = 2,
          "is_sparse"_a   = false,
          "is_precise"_a  = false);
    m.def("discover_composition",
          &discover_composition,
          "A function that discovers differently distributed partitions of the dataset as well "
          "as significant patterns and characterizes the partitions using the maximum entropy "
          "distribution",
          "dataset"_a,
          "min_support"_a = 2,
          "alpha"_a       = 0.05,
          "is_sparse"_a   = false,
          "is_precise"_a  = false);

    for (int i = 0; i <= 1; ++i)
    {
        for (int j = 0; j <= 1; ++j)
        {
            sd::disc::build_trait(
                i, j, [&](auto trait) { declare_py_distribution<decltype(trait)>(m); });
        }
    }

    m.attr("__version__") = "dev";
}
