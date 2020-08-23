#pragma once

#include <bindings/common/TraitBuilder.hxx>
#include <disc/desc/Composition.hxx>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace sd::disc::pyutils
{

template <typename T = std::chrono::milliseconds, typename Fn>
auto timeit(Fn&& fn)
{
    namespace ch = std::chrono;
    // using namespace std::chrono_literals;

    auto before = ch::high_resolution_clock::now();
    std::forward<Fn>(fn)();
    auto after = ch::high_resolution_clock::now();
    // return (after - before) / 1ms;
    return ch::duration_cast<T>(after - before);
}

namespace py = pybind11;

template <typename py_list_t, typename S>
auto& pylist_to_itemset(const py_list_t& x, S& out)
{
    out.clear();
    for (const auto& i : x)
    {
        out.insert(py::cast<size_t>(i));
    }
    return out;
}

template <typename S>
Dataset<S> create_dataset(const py::list& xss)
{
    itemset<S> buffer;
    Dataset<S> out;
    out.reserve(xss.size());
    for (const auto& xs : xss)
    {
        out.insert(pylist_to_itemset(xs, buffer));
    }
    return out;
}

template <typename S>
Dataset<S> create_dataset(const py::array_t<bool>& x)
{
    if (x.ndim() != 2)
        throw std::domain_error{"data is not a matrix"};

    size_t n = x.shape()[0];
    size_t m = x.shape()[1];

    itemset<S> t;
    Dataset<S> out;
    out.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
        t.clear();
        for (size_t j = 0; j < m; ++j)
        {
            if (*x.data(i, j))
                t.insert(j);
        }
        out.insert(t);
    }

    return out;
}

template <typename S>
Dataset<S> create_dataset_pyobject(const py::object& x)
{
    if (py::isinstance<py::list>(x))
    {
        return create_dataset<S>(py::cast<py::list>(x));
    }
    if (py::isinstance<py::array>(x))
    {
        return create_dataset<S>(py::cast<py::array_t<bool>>(x));
    }
    throw std::runtime_error{"cannot represent the given python object as data"};
}

template <typename Data>
py::list data_to_pylist(const Data& in)
{
    py::list out;
    using namespace sd::disc;
    for (const auto& x : in)
    {
        py::list xx;
        sd::foreach(point(x), [&](size_t i) { xx.append(i); });
        out.append(std::move(xx));
    }
    return out;
}

template <typename Trait>
py::dict translate_to_pydict(Component<Trait> const& c, std::chrono::milliseconds ms)
{
    py::list patternset  = data_to_pylist(c.summary);
    py::list frequencies = py::cast(c.summary.template col<0>());
    py::dict r;

    r["pattern_set"] = patternset;
    r["frequencies"] = frequencies;

    auto f0                = c.initial_encoding;
    auto f1                = c.encoding;
    r["initial_objective"] = std::tuple(f0.objective(), f0.of_data, f0.of_model);
    r["objective"]         = std::tuple(f1.objective(), f1.of_data, f1.of_model);
    r["elapsed_time[ms]"]  = ms.count();

    return r;
}

template <typename Trait>
py::dict translate_to_pydict(Composition<Trait> const& c, std::chrono::milliseconds ms)
{
    auto frequencies = py::array_t<double>();
    frequencies.resize({c.frequency.extent(0), c.frequency.extent(1)});
    auto target = static_cast<double*>(frequencies.request().ptr);
    for (size_t i = 0; i < c.frequency.size(); ++i)
    {
        target[i] = static_cast<double>(c.frequency.data()[i]);
    }

    auto A = py::array_t<double>();
    A.resize({c.assignment.size(), c.summary.size() /* - c.data.dim */});

    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        for (auto j : c.assignment[i].container)
        {
            // if (!is_singleton(c.summary.point(j)))
            A.mutable_at(i, j) = 1;
        }
    }

    py::list assignment_list;
    for (const auto& r : c.assignment)
    {
        py::list xx;
        for (auto i : r.container)
        {
            xx.append(i);
        };
        assignment_list.append(std::move(xx));
    }

    py::list labels     = py::cast(c.data.template col<0>());
    py::list patternset = data_to_pylist(c.summary);

    py::dict r;

    r["pattern_set"]     = patternset;
    r["frequencies"]     = frequencies;
    r["assignment_list"] = assignment_list;
    r["assignment"]      = A;
    r["labels"]          = labels;

    auto f0 = c.initial_encoding;
    auto f1 = c.encoding;

    r["initial_objective"] = std::tuple(f0.objective(), f0.of_data, f0.of_model);
    r["objective"]         = std::tuple(f1.objective(), f1.of_data, f1.of_model);

    r["elapsed_time[ms]"] = ms.count();

    return r;
}

template <typename trait_type, typename S>
py::dict desc_impl(Dataset<S> dataset, std::vector<size_t> labels, size_t min_support)
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
        c.data = PartitionedData<S>(std::move(dataset), std::move(labels));
        initialize_model(c, cfg);
        auto initial_encoding = c.encoding = encode(c, cfg);
        auto ms                            = timeit([&] { discover_patterns_generic(c, cfg); });
        c.initial_encoding                 = initial_encoding;
        c.data.revert_order();

        return translate_to_pydict(c, ms);
    }
    else
    {
        Component<trait_type> c;
        c.data  = std::move(dataset);
        auto ms = timeit([&] { discover_patterns_generic(c, cfg); });

        return translate_to_pydict(c, ms);
    }

    return {};
}

template <typename S>
auto desc(Dataset<S>&       dataset,
          std::vector<int>& labels,
          size_t            min_support,
          bool              use_higher_precision_floats = false)
{
    py::dict r;
    sd::disc::select_real_type<S>(use_higher_precision_floats, [&](auto trait) {
        std::vector<size_t> yy(labels.begin(), labels.end());
        r = desc_impl<decltype(trait)>(dataset, yy, min_support);
    });

    return r;
}

template <typename trait_type>
auto disc_impl(Dataset<typename trait_type::pattern_type>&& dataset,
               size_t                                       min_support,
               double                                       alpha)
{
    using namespace sd::disc;

    DiscConfig cfg;
    cfg.alpha            = alpha;
    cfg.min_support      = min_support;
    cfg.use_bic          = true;
    cfg.max_factor_size  = 8;
    cfg.max_factor_width = 10;

    Composition<trait_type> c;
    c.data = PartitionedData<typename trait_type::pattern_type>(std::move(dataset));
    initialize_model(c, cfg);
    auto initial_encoding = c.encoding = encode(c, cfg);
    auto pm = [](auto& c, const auto& g) { discover_patterns_generic(c, g); };

    auto ms = timeit([&] { discover_components(c, cfg, pm, sd::EmptyCallback{}); });

    c.initial_encoding = initial_encoding;
    c.data.revert_order();

    return translate_to_pydict(c, ms);
}

template <typename S>
auto disc(Dataset<S>&       dataset,
          std::vector<int>& labels,
          size_t            min_support,
          bool              use_higher_precision_floats = false)
{
    py::dict r;
    sd::disc::select_real_type<S>(use_higher_precision_floats, [&](auto trait) {
        std::vector<size_t> yy(labels.begin(), labels.end());
        r = disc_impl<decltype(trait)>(dataset, yy, min_support);
    });

    return r;
}

auto describe_partitions(const py::object& dataset,
                         const py::object& labels,
                         size_t            min_support,
                         bool              is_sparse                   = false,
                         bool              use_higher_precision_floats = false)
{
    py::dict r;
    sd::disc::build_trait(is_sparse, use_higher_precision_floats, [&](auto trait) {
        using T = decltype(trait);
        using S = typename T::pattern_type;
        auto x  = create_dataset_pyobject<S>(dataset);
        r       = pyutils::desc_impl<T>(x, py::cast<std::vector<size_t>>(labels), min_support);
    });
    return r;
}

auto discover_composition(const py::object& dataset,
                          size_t            min_support,
                          double            alpha,
                          bool              is_sparse                   = false,
                          bool              use_higher_precision_floats = false)
{
    py::dict r;
    sd::disc::build_trait(is_sparse, use_higher_precision_floats, [&](auto trait) {
        using T = decltype(trait);
        using S = typename T::pattern_type;
        r = pyutils::disc_impl<T>(create_dataset_pyobject<S>(dataset), min_support, alpha);
    });
    return r;
}

} // namespace sd::disc::pyutils
