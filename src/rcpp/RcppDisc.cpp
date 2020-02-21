#include <disc/disc/Disc.hxx>
#include <disc/disc/Desc.hxx>

#include <Rcpp.h>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]

using namespace sd::disc;

using itemset_type = sd::disc::tag_dense;
using float_type = double;
using trait_type = sd::disc::Trait<itemset_type, float_type, sd::disc::MEDistribution<itemset_type, float_type>>;


//' A function that discovers significant patterns for a dataset using the maximum entropy distribution
//' @param dataset in sparse matrix format, i.e. a list of lists. E.g. list(c(1,2), c(2, 3))
//' @param alpha the initial significance level for hypothesis tests
//' @param min_support the minimal support a pattern observed in the dataset in order to be considered as candidate
//' @return the composition: patterns, their frequencies and assignments, BIC scores
//' @export
// [[Rcpp::export]]
Rcpp::List discover_patternset(const Rcpp::List& dataset, double alpha = 0.05, size_t min_support = 1) 
{
    PatternsetResult<trait_type> c;
    itemset<itemset_type> buf;
    for(const auto& xs : dataset) 
    {
        buf.clear();
        for(int x : Rcpp::List(xs)) 
        {
            buf.insert(x);
        }
        c.data.insert(buf);
    }

    MiningSettings cfg;
    cfg.alpha = alpha;
    cfg.min_support = min_support;
    cfg.use_bic = true;

    c = discover_patternset(std::move(c), cfg);
    
    Rcpp::List result;

    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for(auto & x : c.summary.col<1>()) {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        iterate_over(x, [&](size_t i) { b.push_back(i); });
    } 
    result["patternset"] = patternset;
    result["BIC0"] = c.initial_encoding.objective();
    result["BIC"] = c.encoding.objective();
    result["frequencies"] = c.summary.col<0>().container();

    return result;
}

//' A function that discovers significant patterns and characterizes multiple, given a partition using the maximum entropy distribution
//' @param dataset in sparse matrix format, i.e. a list of lists. E.g. list(c(1,2), c(2, 3))
//' @param labels a list of labels for each data-point as indicator for partitions
//' @param alpha the initial significance level for hypothesis tests
//' @param min_support the minimal support a pattern observed in the dataset in order to be considered as candidate
//' @return the composition: patterns, their frequencies and assignments, BIC scores
//' @export
// [[Rcpp::export]]
Rcpp::List characterize_partitions(const Rcpp::List& dataset, const Rcpp::List& labels, double alpha = 0.05, size_t min_support = 1) 
{
    Composition<trait_type> c;
    itemset<itemset_type> buf;
    size_t i = 0;
    for(const auto& xs : dataset) 
    {
        buf.clear();
        for(int x : Rcpp::List(xs)) 
        {
            buf.insert(x);
        }
        c.data.insert(buf, static_cast<size_t>(labels[i]));
        ++i;
    }

    MiningSettings cfg;
    cfg.alpha = alpha;
    cfg.min_support = min_support;
    cfg.use_bic = true;

    initialize_composition(c, cfg);
    c = discover_patternsets(std::move(c), cfg);
    
    Rcpp::List result;

    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for(auto & x : c.summary.col<1>()) {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        iterate_over(x, [&](size_t i) { b.push_back(i); });
    } 

    std::vector<std::vector<sparse_index_type>> assignment;
    sd::sparse_dynamic_bitset<sparse_index_type> spbuf;
    for(auto && a : c.assignment) {
        spbuf.insert(a);
        assignment.emplace_back(spbuf.container);
        spbuf.clear();
    }

    result["patternset"] = patternset;
    result["assignment"] = assignment;
    result["BIC0"] = c.initial_encoding.objective();
    result["BIC"] = c.encoding.objective();
    result["frequencies"] = Rcpp::NumericMatrix(
        c.frequency.extent(0), 
        c.frequency.extent(1), 
        c.frequency.data()
    );

    return result;
}



//' A function that discovers differently distributed partitions of the dataset as well as significant patterns and characterizes the partitions using the maximum entropy distribution
//' @param dataset in sparse matrix format, i.e. a list of lists. E.g. list(c(1,2), c(2, 3))
//' @param alpha the initial significance level for hypothesis tests
//' @param min_support the minimal support a pattern observed in the dataset in order to be considered as candidate
//' @return the composition: partition-labels, patterns, their frequencies and assignments, BIC scores
//' @export
// [[Rcpp::export]]
Rcpp::List discover_composition(const Rcpp::List& dataset, double alpha = 0.05, size_t min_support = 1) 
{
    Composition<trait_type> c;
    itemset<itemset_type> buf;
    for(const auto& xs : dataset) 
    {
        buf.clear();
        for(int x : Rcpp::List(xs)) 
        {
            buf.insert(x);
        }
        c.data.insert(buf);
    }

    DecompositionSettings cfg;
    cfg.alpha = alpha;
    cfg.min_support = min_support;
    cfg.use_bic = true;

    initialize_composition(c, cfg);
    c = mine_split_round_repeat(std::move(c), cfg);
    
    Rcpp::List result;

    std::vector<std::vector<size_t>> patternset;

    patternset.reserve(c.summary.size());

    for(auto & x : c.summary.col<1>()) {
        auto& b = patternset.emplace_back();
        b.reserve(count(x));
        iterate_over(x, [&](size_t i) { b.push_back(i); });
    } 

    std::vector<std::vector<sparse_index_type>> assignment;
    sd::sparse_dynamic_bitset<sparse_index_type> spbuf;
    for(auto && a : c.assignment) {
        spbuf.insert(a);
        assignment.emplace_back(spbuf.container);
        spbuf.clear();
    }

    result["patternset"] = patternset;
    result["assignment"] = assignment;
    result["labels"] = c.data.col<0>().container();
    result["BIC0"] = c.initial_encoding.objective();
    result["BIC"] = c.encoding.objective();
    result["frequencies"] = Rcpp::NumericMatrix(
        c.frequency.extent(0), 
        c.frequency.extent(1), 
        c.frequency.data()
    );

    return result;
}

struct RMEDistribution : public sd::disc::MEDistribution<itemset_type, float_type> {
    using base = sd::disc::MEDistribution<itemset_type, float_type>;
    
    RMEDistribution(size_t dim) : base(dim, 0) {
        estimate_model(*this);
    }

     void insert(float_type label, const Rcpp::List& t)
    {
        buf.clear();
        for(size_t i : t) {
            buf.insert(i);
        }

        base::insert(label, buf);
        sd::disc::estimate_model(*this);
    }

    void insert_batch(const Rcpp::List& ys, const Rcpp::List& ts)
    {
        for(size_t i = 0; i < ts.size(); ++i) {
            buf.clear();
            for(int j : Rcpp::List(ts[i])) {
                buf.insert(j);
            }
            base::insert(static_cast<float_type>(ys[i]), buf);
        }
        sd::disc::estimate_model(*this);
    }

    float_type infer_generalized_itemset(const Rcpp::List& t) const
    {
        thread_local sd::disc::itemset<itemset_type> buf;
        buf.clear();
        for(size_t i : t) {
            buf.insert(i);
        }
        return base::expected_generalized_frequency(buf);
    }

    float_type infer(const Rcpp::List& t) const
    {
        thread_local sd::disc::itemset<itemset_type> buf;
        buf.clear();
        for(size_t i : t) {
            buf.insert(i);
        }
        return base::expected_frequency(buf);
    }
private:
    sd::disc::itemset<itemset_type> buf;
};


//' Create instances of the Maximum Entropy Distribution from DISC using the underlying C++ implementation
//'
//' @param dim Number of Dimensions of the Dataset
//'
//' @return
//' A `MaxEnt Distribution` object
//'
//' @examples
//'
//' p = new(MEDistribution, 10)
//'
//' p$infer(c(1,2))
//'
//' p$insert(0.1, c(1,2))
//' p$infer(c(1,2))
//'
//' @useDynLib discminer, .registration = TRUE
//' @import methods, Rcpp
//' @importFrom Rcpp, sourceCPP
//' @name MEDistribution
//' @export MEDistribution
RCPP_MODULE(MEDistribution) {
    Rcpp::class_<RMEDistribution>("MEDistribution")
        .constructor<size_t>()
        .method("infer", &RMEDistribution::infer, "infer a probability of an itemset")
        .method("infer_generalized_itemset", &RMEDistribution::infer_generalized_itemset, "infer the frequency of an generalized itemset")
        .method("insert_batch", &RMEDistribution::insert_batch, "add a list of constraints (patterns and frequencies) to the distribution, then estimate coefficients")
        .method("insert", &RMEDistribution::insert, "add one additional constraint (pattern and frequency) to the distribution, then estimate coefficients");
}