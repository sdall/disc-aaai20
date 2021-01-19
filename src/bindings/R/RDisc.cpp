#include <Rcpp.h>

#include <bindings/common/TraitBuilder.hxx>
#include <disc/desc/Desc.hxx>
#include <disc/disc/Disc.hxx>
#include <disc/utilities/ModelPruning.hxx>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]

using namespace sd::disc;


struct IDescMK2 : sd::disc::DefaultPatternsetMinerInterface
{
    template <typename C, typename Config>
    static auto finish(C& c, const Config& cfg)
    {
        sd::disc::prune_pattern_composition(c, cfg);
    }
};


template <typename trait_type>
Rcpp::List translate_to_rcpp(Component<trait_type> const& c, BiMap& tr)
{
    Rcpp::List                       result;
    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for (auto& x : c.summary.template col<1>())
    {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        foreach(x, [&](size_t i) { b.push_back(tr.convert_to_external(i)); });
    }
    result["patternset"]        = patternset;
    result["initial_objective"] = c.initial_encoding.objective();
    result["objective"]         = c.encoding.objective();
    result["frequencies"]       = c.summary.template col<0>();
    return result;
}

template <typename S>
auto create_dataset(const Rcpp::List& dataset, BiMap& tr)
{
    Dataset<S> out;
    itemset<S> buf;
    for (const auto& xs : dataset)
    {
        buf.clear();
        for (size_t x : Rcpp::List(xs))
        {
            buf.insert(tr.convert_to_internal(x));
        }
        out.insert(buf);
    }
    return out;
}

template <typename T>
void create_dataset(const Rcpp::List& dataset, Component<T>& c)
{
    c.data = create_dataset<typename T::pattern_type>(dataset);
}
template <typename T>
void create_dataset(const Rcpp::List& x, const Rcpp::List& y, Composition<T>& c, BiMap& tr)
{
    using S = typename T::pattern_type;
    if (y.size() > 0)
    {
        std::vector<size_t> yy(y.size());
        std::copy(y.begin(), y.end(), yy.begin()); // casting!
        c.data = PartitionedData<S>(create_dataset<S>(x, tr), yy);
    }
    else
        c.data = PartitionedData<S>(create_dataset<S>(x, tr));
}

template <typename trait_type>
Rcpp::List translate_to_rcpp_list(Composition<trait_type> const& c, BiMap& tr)
{
    Rcpp::List result;

    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for (auto& x : c.summary.template col<1>())
    {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        foreach(x, [&](size_t i) { b.push_back(tr.convert_to_external(i)); });
    }

    std::vector<std::vector<sparse_index_type>>  assignment;
    sd::sparse_dynamic_bitset<sparse_index_type> spbuf;
    for (auto&& a : c.assignment)
    {
        spbuf.insert(a);
        assignment.emplace_back(spbuf.container);
        spbuf.clear();
    }

    result["patternset"]        = patternset;
    result["assignment"]        = assignment;
    result["initial_objective"] = c.initial_encoding.objective();
    result["objective"]         = c.encoding.objective();
    result["frequencies"] =
        Rcpp::NumericMatrix(c.frequency.extent(0), c.frequency.extent(1), c.frequency.data());

    return result;
}

template <typename trait_type>
Rcpp::List
discover_patternset(const Rcpp::List& dataset, const Rcpp::List& labels, size_t min_support)
{
    DiscConfig cfg;
    cfg.min_support = min_support;
    cfg.use_bic     = true;

    BiMap tr;

    if (labels.size() == 0)
    {
        Component<trait_type> c;
        create_dataset(dataset, c, tr);
        sd::disc::discover_patterns_generic(c, cfg, IDescMK2{});
        return translate_to_rcpp_list(c, tr);
    }
    else
    {
        Composition<trait_type> c;
        create_dataset(dataset, labels, c);
        initialize_model(c, cfg);
        auto initial_encoding = c.encoding = encode(c, cfg);
        discover_patterns_generic(c, cfg, IDescMK2{});
        c.initial_encoding = initial_encoding;
        c.data.revert_order();
        return translate_to_rcpp_list(c, tr);
    }

    return {};
}

///' A function that discovers significant patterns for a dataset using the maximum entropy distribution
///' @param dataset in sparse matrix format, i.e. a list of lists. E.g. list(c(1,2), c(2, 3)) 
///' @param alpha the initial significance level for hypothesis tests
///' @param min_support the minimal support a pattern observed in the dataset in order to beconsidered as candidate
///' @return the composition: patterns, their frequencies and assignments, BIC scores

// [[Rcpp::export]]
Rcpp::List desc(const Rcpp::List& dataset,
                const Rcpp::List& labels                      = R_NilValue,
                size_t            min_support                 = 1,
                bool              is_sparse                   = false,
                bool              use_higher_precision_floats = false)
{
    Rcpp::List result;
    build_trait(is_sparse, use_higher_precision_floats, [&](auto trait) {
        result = discover_patternset<decltype(trait)>(dataset, labels, min_support);
    });
    return result;
}

template <typename trait_type>
Rcpp::List
discover_composition(const Rcpp::List& dataset, double alpha = 0.05, size_t min_support = 1)
{
    DiscConfig cfg;
    cfg.alpha       = alpha;
    cfg.min_support = min_support;
    cfg.use_bic     = true;
    BiMap tr;
    Composition<trait_type> c;
    create_dataset(dataset, {}, c, tr);
    initialize_model(c, cfg);
    auto initial_encoding = c.encoding = encode(c, cfg);

    auto pm = [](auto& c, const auto& g) { discover_patterns_generic(c, g, IDescMK2{}); };
    discover_components(c, cfg, pm, sd::EmptyCallback{});
    c.initial_encoding = initial_encoding;
    c.data.revert_order();
    return translate_to_rcpp_list(c, tr);
}

///' A function that discovers differently distributed partitions of the dataset as well as
/// significant patterns and characterizes the partitions using the maximum entropy distribution
///' @param dataset in sparse matrix format, i.e. a list of lists. E.g. list(c(1,2), c(2, 3))
///' @param alpha the initial significance level for hypothesis tests
///' @param min_support the minimal support a pattern observed in the dataset in order to be
///' considered as candidate
///' @return the composition: partition-labels, patterns, their
///' frequencies and assignments, BIC scores

// [[Rcpp::export]]
Rcpp::List disc(const Rcpp::List& dataset,
                double            alpha                       = 0.05,
                size_t            min_support                 = 1,
                bool              is_sparse                   = false,
                bool              use_higher_precision_floats = false)
{
    Rcpp::List result;
    
    build_trait(is_sparse, use_higher_precision_floats, [&](auto trait) {
        result = discover_composition<decltype(trait)>(dataset, alpha, min_support);
    });
    return result;
}
