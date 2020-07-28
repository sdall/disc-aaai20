#include <Rcpp.h>

#include <disc/desc/Desc.hxx>
#include <disc/disc/Disc.hxx>
#include <utils/TraitBuilder.hxx>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]

using namespace sd::disc;

template <typename trait_type>
Rcpp::List translate_to_rcpp(Component<trait_type> const& c)
{
    Rcpp::List                       result;
    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for (auto& x : c.summary.template col<1>())
    {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        foreach(x, [&](size_t i) { b.push_back(i); });
    }
    result["patternset"]  = patternset;
    result["BIC0"]        = c.initial_encoding.objective();
    result["BIC"]         = c.encoding.objective();
    result["frequencies"] = c.summary.template col<0>();
    return result;
}

template <typename Data>
auto create_dataset(const Rcpp::List& dataset, Data& out)
{
    itemset<typename Data::pattern_type> buf;
    for (const auto& xs : dataset)
    {
        buf.clear();
        for (int x : Rcpp::List(xs))
        {
            buf.insert(x);
        }
        out.insert(buf);
    }
}

template <typename T>
void create_dataset(const Rcpp::List& dataset, Component<T>& c)
{
    c.data = create_dataset<typename T::pattern_type>(dataset);
}
template <typename T>
void create_dataset(const Rcpp::List& dataset, Composition<T>& c)
{
    using S = typename T::pattern_type;
    c.data = PartitionedData<S>(create_dataset<T>(dataset));
}
template <typename T>
void create_dataset(const Rcpp::List& dataset,const Rcpp::List& labels, Composition<T>& c)
{
    using S = typename T::pattern_type;
    std::vector<size_t> y = labels;
    // for(size_t l : labels) y.push_back(l);
    c.data = PartitionedData<S>(create_dataset<T>(dataset), y);
}

// template <typename Data>
// auto insert_rcpp_to(const Rcpp::List& dataset, const Rcpp::List& labels, Data& out)
// {
//     itemset<typename Data::pattern_type> buf;
//     size_t                               i = 0;
//     for (const auto& xs : dataset)
//     {
//         buf.clear();
//         for (int x : Rcpp::List(xs))
//         {
//             buf.insert(x);
//         }
//         out.insert(buf, static_cast<size_t>(labels[i]));
//         ++i;
//     }
// }

template <typename trait_type>
Rcpp::List translate_to_rcpp_list(Composition<trait_type> const& c)
{
    Rcpp::List result;

    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for (auto& x : c.summary.template col<1>())
    {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        foreach(x, [&](size_t i) { b.push_back(i); });
    }

    std::vector<std::vector<sparse_index_type>>  assignment;
    sd::sparse_dynamic_bitset<sparse_index_type> spbuf;
    for (auto&& a : c.assignment)
    {
        spbuf.insert(a);
        assignment.emplace_back(spbuf.container);
        spbuf.clear();
    }

    result["patternset"] = patternset;
    result["assignment"] = assignment;
    result["BIC0"]       = c.initial_encoding.objective();
    result["BIC"]        = c.encoding.objective();
    result["frequencies"] =
        Rcpp::NumericMatrix(c.frequency.extent(0), c.frequency.extent(1), c.frequency.data());

    return result;
}

template <typename trait_type>
Rcpp::List discover_patternset_(const Rcpp::List& dataset, size_t min_support = 1)
{

    DiscConfig cfg;
    cfg.min_support = min_support;
    cfg.use_bic     = true;

    Component<trait_type> c;
    create_dataset(dataset, c);

    sd::disc::discover_patterns_generic(c, cfg);
    return translate_to_rcpp(c);
}

//' A function that discovers significant patterns for a dataset using the maximum entropy
// distribution ' @param dataset in sparse matrix format, i.e. a list of lists. E.g.
// list(c(1,2), c(2, 3)) ' @param alpha the initial significance level for hypothesis tests '
// @param min_support the minimal support a pattern observed in the dataset in order to be
// considered as candidate ' @return the composition: patterns, their frequencies and
// assignments, BIC scores ' @export
// [[Rcpp::export]]
Rcpp::List discover_patternset(const Rcpp::List& dataset,
                               const Rcpp::List& labels,
                               size_t            min_support = 1,
                               bool              is_sparse   = false,
                               bool              is_precise  = false)
{
    Rcpp::List result;
    build_trait(is_sparse, is_precise, [&](auto trait) {
        result = discover_patternset_<decltype(trait)>(dataset, min_support);
    });
    return result;
}

template <typename trait_type>
Rcpp::List characterize_partitions_(const Rcpp::List& dataset,
                                    const Rcpp::List& labels,
                                    size_t            min_support = 1)
{
    DiscConfig cfg;
    cfg.min_support = min_support;
    cfg.use_bic     = true;

    Composition<trait_type>                    c;
    create_dataset(dataset, labels, c);
    initialize_model(c, cfg);
    auto initial_encoding = co.encoding = encode(co, cfg);

    discover_patterns_generic(c, cfg);
    co.initial_encoding = initial_encoding;

    return translate_to_rcpp_list(c);
}

//' A function that discovers significant patterns and characterizes multiple, given a partition
// using the maximum entropy distribution ' @param dataset in sparse matrix format, i.e. a list
// of lists. E.g. list(c(1,2), c(2, 3)) ' @param labels a list of labels for each data-point as
// indicator for partitions ' @param alpha the initial significance level for hypothesis tests '
//@param min_support the minimal support a pattern observed in the dataset in order to be
// considered as candidate ' @return the composition: patterns, their frequencies and
// assignments, BIC scores ' @export
// [[Rcpp::export]]
Rcpp::List characterize_partitions(const Rcpp::List& dataset,
                                   const Rcpp::List& labels,
                                   size_t            min_support = 1,
                                   bool              is_sparse   = false,
                                   bool              is_precise  = false)
{
    Rcpp::List result;
    build_trait(is_sparse, is_precise, [&](auto trait) {
        result = characterize_partitions_<decltype(trait)>(dataset, labels, min_support);
    });
    return result;
}

template <typename trait_type>
Rcpp::List
discover_composition_(const Rcpp::List& dataset, double alpha = 0.05, size_t min_support = 1)
{
    DiscConfig cfg;
    cfg.alpha       = alpha;
    cfg.min_support = min_support;
    cfg.use_bic     = true;
    
    Composition<trait_type>                    c;
    create_dataset(dataset, labels, c);
    initialize_model(c, cfg);
    auto initial_encoding = co.encoding = encode(co, cfg);
    
    auto pm = [](auto& c, const auto& g) { discover_patterns_generic(c, g); };
    discover_components(c, cfg, pm, sd::EmptyCallback{});
    co.initial_encoding = initial_encoding;

    return translate_to_rcpp_list(c);
}

//' A function that discovers differently distributed partitions of the dataset as well as
// significant patterns and characterizes the partitions using the maximum entropy distribution
//' @param dataset in sparse matrix format, i.e. a list of lists. E.g. list(c(1,2), c(2, 3))
//' @param alpha the initial significance level for hypothesis tests
//' @param min_support the minimal support a pattern observed in the dataset in order to be
// considered as candidate ' @return the composition: partition-labels, patterns, their
// frequencies and assignments, BIC scores ' @export
// [[Rcpp::export]]
Rcpp::List discover_composition(const Rcpp::List& dataset,
                                double            alpha       = 0.05,
                                size_t            min_support = 1,
                                bool              is_sparse   = false,
                                bool              is_precise  = false)
{
    Rcpp::List result;
    build_trait(is_sparse, is_precise, [&](auto trait) {
        result = discover_composition_<decltype(trait)>(dataset, alpha, min_support);
    });
    return result;
}

template <typename S, typename T>
struct RMEDist
{
    sd::disc::MaxEntDistribution<S, T> dist;
    using base = sd::disc::MaxEntDistribution<S, T>;

    // RMEDist(size_t dim) : dist(dim, 1) { estimate_model(dist); }

    RMEDist(size_t dim, size_t max_factor_size = 8, size_t max_factor_width = 12)
        : dist(dim, 1, max_factor_size, max_factor_width)
    {
        sd::disc::estimate_model(dist);
    }

    void insert(T label, const Rcpp::List& t)
    {
        buf.clear();
        for (size_t i : t)
        {
            buf.insert(i);
        }

        dist.insert(label, buf, true);
    }

    void insert_batch(const Rcpp::List& ys, const Rcpp::List& ts)
    {
        for (size_t i = 0; i < ts.size(); ++i)
        {
            buf.clear();
            for (int j : Rcpp::List(ts[i]))
            {
                buf.insert(j);
            }
            dist.insert(static_cast<T>(ys[i]), buf, false);
        }
        sd::disc::estimate_model(dist);
    }

    T infer_generalized_itemset(const Rcpp::List& t) const
    {
        thread_local sd::disc::itemset<S> buf;
        buf.clear();
        for (size_t i : t)
        {
            buf.insert(i);
        }
        return dist.expectation_generalized_set(buf);
    }

    T infer(const Rcpp::List& t) const
    {
        thread_local sd::disc::itemset<S> buf;
        buf.clear();
        for (size_t i : t)
        {
            buf.insert(i);
        }
        return dist.expectation(buf);
    }

    static std::string type_name()
    {
        std::string name = "MEDist_";
        name += storage_type_to_str<S>();
        name += "_";
        name += float_storage_type_to_str<T>();
        return name;
    }

private:
    sd::disc::itemset<S> buf;
};
//' Create instances of the Maximum Entropy Distribution from DISC using the underlying C++
// implementation
//'
//' @param dim Number of Dimensions of the Dataset
//'
//' @return
//' A `MaxEnt Distribution` object
//'
//' @examples
//'
//' p = new(MaxEntDistribution, 10)
//'
//' p$infer(c(1,2))
//'
//' p$insert(0.1, c(1,2))
//' p$infer(c(1,2))
//'
//' @useDynLib discminer, .registration = TRUE
//' @import methods, Rcpp
//' @importFrom Rcpp, sourceCPP
//' @name MaxEntDistribution##_##TAG##_##FLOAT
//' @export MaxEntDistribution##_##TAG##_##FLOAT

const char* info_infer     = "infer a probability of an itemset";
const char* info_gen_infer = "infer the frequency of an generalized itemset";
const char* info_insert_b  = "add a list of constraints (patterns and frequencies) to the "
                            "distribution, then estimate coefficients";
const char* info_insert = "add one additional constraint (pattern and frequency) to the "
                          "distribution, then estimate coefficients";

#define DEF_ME_DIST(Class)                                                                     \
    /* [[Rcpp::export]] */                                                                     \
    Rcpp::class_<Class>(Class::type_name().c_str())                                            \
        .constructor<std::size_t>()                                                            \
        .constructor<std::size_t, std::size_t>()                                               \
        .constructor<std::size_t, std::size_t, std::size_t>()                                  \
        .method("infer", &Class::infer, info_infer)                                            \
        .method("infer_generalized", &Class::infer_generalized_itemset, info_gen_infer)        \
        .method("insert_batch", &Class::insert_batch, info_insert_b)                           \
        .method("insert", &Class::insert, info_insert);

// #define MAKE_RCPP_ME_DIST(TAG, FLOAT)                                                          \
//     using ME##_##TAG##_##FLOAT = RMEDist<tag##_##TAG, FLOAT>;                          \
//     RCPP_EXPOSED_CLASS_NODECL(ME##_##TAG##_##FLOAT)                                            \
//     RCPP_MODULE(ME##_##TAG##_##FLOAT) { DEF_ME_DIST(ME##_##TAG##_##FLOAT) }

// MAKE_RCPP_ME_DIST(dense, double)
// MAKE_RCPP_ME_DIST(dense, precise_float_t)
// MAKE_RCPP_ME_DIST(sparse, double)
// MAKE_RCPP_ME_DIST(sparse, precise_float_t)
// #undef MAKE_RCPP_ME_DIST

using ME_dense_double = RMEDist<tag_dense, double>;
RCPP_EXPOSED_CLASS_NODECL(ME_dense_double)
RCPP_MODULE(ME_dense_double) { DEF_ME_DIST(ME_dense_double) }

using ME_sparse_double = RMEDist<tag_sparse, double>;
RCPP_EXPOSED_CLASS_NODECL(ME_sparse_double)
RCPP_MODULE(ME_sparse_double) { DEF_ME_DIST(ME_sparse_double) }

using ME_dense_ldouble = RMEDist<tag_dense, precise_float_t>;
RCPP_EXPOSED_CLASS_NODECL(ME_dense_ldouble)
RCPP_MODULE(ME_dense_ldouble) { DEF_ME_DIST(ME_dense_ldouble) }

using ME_sparse_ldouble = RMEDist<tag_sparse, precise_float_t>;
RCPP_EXPOSED_CLASS_NODECL(ME_sparse_ldouble)
RCPP_MODULE(ME_sparse_ldouble) { DEF_ME_DIST(ME_sparse_ldouble) }
