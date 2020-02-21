#pragma once

#include <mtv/expmodelgraph.h>
#include <mtv/summaryminer.h>
// #include <mtv/trie.h>

#include <disc/disc/Encoding.hxx>
#include <disc/disc/InsertMissingSingletons.hxx>
#include <disc/distribution/Distribution.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/storage/Itemset.hxx>
#include <disc/utilities/MeasureElapsedTime.hxx>

#include <containers/random-access-set.hxx>
#include <trie/Trie.hxx>

#include <chrono>
#include <map>
#include <vector>

namespace sd
{
namespace disc
{

template <typename T, typename S>
static T save_narrow_cast(S&& s)
{
    // std::decay_t<S> ss = s;
    bool sign = (s < std::decay_t<S>{});
    T    t    = static_cast<T>(s);
    if (s != static_cast<std::decay_t<S>>(t) || sign != (t < std::decay_t<T>{}))
        throw std::runtime_error{"unsave narrowing cast"};
    return t;
}

template <typename X>
void push_back_into_mtv_itemset(const X& x, std::vector<int>& y)
{
    iterate_over(x, [&y](auto i) {
        assert(i <= std::numeric_limits<int>::max());
        y.push_back(save_narrow_cast<int>(i));
    });
}

template <typename X>
void assign_to_mtv_itemset(const X& x, std::vector<int>& y)
{
    y.clear();
    push_back_into_mtv_itemset(x, y);
}

struct MTVModel
{
    constexpr static double const epsilon{1e-9};
    using value_type = double;

    size_t max_num_itemsets = 5;
    size_t max_range_size   = 8;

    MTVModel(size_t dimension, size_t data_size)
        : mtv_model_ptr{
              std::make_unique<ExpModelGraph>(save_narrow_cast<unsigned int>(dimension),
                                              save_narrow_cast<unsigned int>(data_size))}
    {
        mtv_model().include_items      = true;
        mtv_model().include_rowmargins = false;
    }

    // MEDistribution<S, T>(const MEDistribution<S, T>&) = delete;
    // MEDistribution<S, T>& operator=(const MEDistribution<S, T>&) = delete;
    // ~MEDistribution<S, T>()                          = default;
    // MEDistribution<S, T>(MEDistribution<S, T>&& rhs)             = default;
    // MEDistribution<S, T>& operator=(MEDistribution<S, T>&&) = default;

    // MEDistribution<S, T>(MEDistribution<S, T>&& rhs) :
    // mtv_model()(std::move(rhs.mtv_model())) {
    //     rhs.clear();
    // }

    // MEDistribution<S, T>& operator=(MEDistribution<S, T>&& rhs) {
    //     mtv_model() = std::move(rhs.mtv_model());
    //     rhs.clear();
    //     return *this;
    // }

    size_t dimension() const { return mtv_model().item_cnt; }
    size_t num_itemsets() const { return mtv_model().all_itemsets_size; }

    template <typename value_type, typename pattern_type>
    void insert(value_type label, const pattern_type& t)
    {
        if (count(t) == 0)
            return;
        // if label is 0 or 1 MTV breaks!
        label = std::clamp<value_type>(label, epsilon, value_type(1) - epsilon);
        assert(label > 0 && label < 1);
        assign_to_mtv_itemset(t, buffer);

#ifndef NDEBUG
        for (auto i : buffer)
            assert((size_t)i < mtv_model().item_cnt);
#endif

        if (buffer.size() == 1)
        {
            mtv_model().set_item_frequency(buffer.front(), label);
        }
        else
        {
            assert(buffer.size() > 1);
            [[maybe_unused]] const auto prev_size = mtv_model().all_itemsets_size;
            mtv_model().add_itemset(buffer.data(), buffer.size(), static_cast<double>(label));
            assert(mtv_model().all_itemsets_size == prev_size + 1);
        }
    }

    template <typename value_type>
    void insert_singleton(value_type label, size_t item)
    {
        assert(item < mtv_model().item_cnt);
        label = std::clamp<value_type>(label, epsilon, value_type(1) - epsilon);
        assert(label > 0 && label < 1);
        mtv_model().set_item_frequency(save_narrow_cast<int>(item), label);
    }

    template <typename value_type, typename pattern_type>
    void insert_singleton(value_type label, const pattern_type& t)
    {
        assert(count(t) == 1);
        insert_singleton(label, front(t));
    }

    template <typename pattern_type>
    auto expected_frequency(const pattern_type& t)
    {
        assign_to_mtv_itemset(t, buffer);
        return mtv_model().estimate_frequency(buffer.data(), buffer.size());
    }

    // auto generalized_expected_frequency(const sd::cpslice<const disc::index_type>& t)
    // {
    //     auto p = probability(t);
    //     for (size_t i = 0; i < mtv_model().item_cnt; ++i)
    //     {
    //         if (mtv_model().item_nodes[i] == 0) continue;
    //         int singleton[1] = {static_cast<int>(i)};
    //         if (!is_subset(i, t))
    //             p *= 1.0 - mtv_model().estimate_frequency(singleton, 1);
    //     }
    //     return p;
    // }

    template <typename S>
    auto generalized_expected_frequency(const base_bitset_sparse<S>& t)
    {
        auto p = probability(t);
        for (size_t i = 0; i < mtv_model().item_cnt; ++i)
        {
            if (mtv_model().item_nodes[i] == 0)
                continue;
            int singleton[1] = {static_cast<int>(i)};
            if (!is_subset(i, t))
                p *= 1.0 - mtv_model().estimate_frequency(singleton, 1);
        }
        return p;
    }

    template <typename S>
    auto generalized_expected_frequency(const sd::bit_view<S>& t)
    {
        auto p = probability(t);
        for (std::uint32_t i = 0; i < mtv_model().item_cnt; ++i)
        {
            int singleton[1] = {static_cast<int>(i)};
            if (!is_subset(i, t))
                p *= 1.0 - mtv_model().estimate_frequency(singleton, 1);
        }
        //  iterate_over(t, [&](auto i) { p /= 1.0 - model.expected_frequency(insert(bits,
        // i));
        // }); for(size_t i = 0; i < dim; ++i) {
        //     p *= 1.0 - model.expected_frequency(insert(bits, i));
        // }
        return p;
    }

    template <typename pattern_type>
    bool is_item_allowed(const pattern_type& t)
    {
        if (max_range_size == 0 && max_num_itemsets == 0)
            return true;

        assign_to_mtv_itemset(t, buffer);
        const auto& itemset = buffer;

        if (itemset.size() > max_range_size)
            return false;
        groups.clear();
        unsigned int node_index = 0;
        for (auto item : itemset)
        {
            if (mtv_model().item_nodes[item])
            {
                groups.insert(mtv_model().item_nodes[item]);
                if (node_index)
                    node_index = std::min(mtv_model().item_nodes[item], node_index);
                else
                    node_index = mtv_model().item_nodes[item];
            }
        }

        if (node_index == 0)
            return true;

        // const auto partition_size = mtv_model().nodes[node_index]->partition_size;
        //     const auto max_partition_size =
        //     mtv_model().nodes[node_index]->max_partition_size; if( (partition_size << 1) >=
        //     max_partition_size) return false;

        size_t len   = 0;
        size_t width = 0;

        for (auto index : groups)
        {
            len += mtv_model().nodes[index]->itemsets_count;
            width += mtv_model().nodes[index]->itemsets_union_size;
        }
        return (!max_range_size || width <= max_range_size) &&
               (!max_num_itemsets || len <= max_num_itemsets);

        // auto not_allowed = ((max_range_size && width > max_range_size) ||
        //     (max_num_itemsets && len + 1 > max_num_itemsets));

        // return !not_allowed;

        // groups.clear();

        // size_t len   = 0;
        // size_t width = 0;
        // unsigned int node_index = 0;
        // for (size_t i = 0; i < itemset.size(); ++i) {
        //     if (mtv_model().item_nodes[itemset[i]] != 0) {
        //         groups.insert(mtv_model().item_nodes[itemset[i]]);
        //         if (node_index)
        //             node_index = std::min(mtv_model().item_nodes[itemset[i]], node_index);
        //         else
        //             node_index = mtv_model().item_nodes[itemset[i]];
        //     }
        //     else
        //         width++;
        // }

        // if (node_index == 0) return true;

        // len   += mtv_model().nodes[node_index]->itemsets_count;
        // width += mtv_model().nodes[node_index]->itemsets_union_size;
        // groups.erase(node_index);
        // for (auto index : groups) {
        //     len += mtv_model().nodes[index]->itemsets_count;
        //     width += mtv_model().nodes[index]->itemsets_union_size;
        // }
        return (!max_range_size || width < max_range_size) &&
               (!max_num_itemsets || len < max_num_itemsets);
    }

    size_t size() const { return mtv_model().all_itemsets_size + mtv_model().item_cnt; }

    ExpModelGraph&       mtv_model() { return *mtv_model_ptr; }
    const ExpModelGraph& mtv_model() const { return *mtv_model_ptr; }

    std::unique_ptr<ExpModelGraph>        mtv_model_ptr;
    std::vector<int>                      buffer;
    andres::RandomAccessSet<unsigned int> groups;

private:
    // void clear()
    // {
    //     mtv_model().node_count        = 0;
    //     mtv_model().nodes             = nullptr;
    //     mtv_model().all_itemsets_size = 0;
    //     mtv_model().all_itemsets      = nullptr;

    //     mtv_model().all_itemsets     = 0;
    //     mtv_model().frequencies      = 0;
    //     mtv_model().item_frequencies = 0;
    //     mtv_model().rowsize_probs    = 0;
    //     mtv_model().item_nodes       = 0;

    //     mpfr_clear(mtv_model().estimate);
    //     mpfr_clear(mtv_model().f);
    //     mpfr_clear(mtv_model().h);
    //     mtv_model().projectedset = nullptr;
    //     mtv_model().thegroups    = nullptr;
    // }
};

auto log_likelihood(MTVModel& model, size_t data_size)
{
    return data_size * model.mtv_model().get_entropy(); // see MTV paper.
}

template <typename Data>
auto log_likelihood(MTVModel& model, const Data& data)
{
    return log_likelihood(model, data.size());
}

void estimate_model(MTVModel& model, size_t max_iterations = 100)
{
    return model.mtv_model().iterative_scaling(max_iterations);
}

struct MTVProbabilityDistribution
{
    template <typename pattern_type>
    auto operator()(MTVModel& model, const pattern_type& t)
    {
        return model.expected_frequency(t);
    }
};

enum class MTVModelCost
{
    BIC = 1,
    MDL = 3 // 2 == some kind of MDL that is not in use internally, I guess.
};

struct MTVMiner
{
    template <typename X>
    auto discover_patternset(const X& data)
    {
        SummaryMiner miner = SummaryMiner{verbose, save_narrow_cast<int>(top_k_queue_size)};
        init_miner(miner);
        import_data(data, miner);
        miner.mine_summary(
            max_items, early_stopping, max_time.count(), ondemand_candidate_generation);
        miner.summary->iterative_scaling();
        return double(data.size()) * double(miner.summary->get_entropy());
    }

    template <typename X, typename OutputData>
    auto discover_patternset(const X& data, OutputData& out)
    {
        SummaryMiner miner = SummaryMiner{verbose, save_narrow_cast<int>(top_k_queue_size)};
        init_miner(miner);
        import_data(data, miner);
        miner.mine_summary(
            max_items, early_stopping, max_time.count(), ondemand_candidate_generation);
        miner.summary->iterative_scaling();
        export_summary(miner, out);
        return double(data.size()) * double(miner.summary->get_entropy());
    }

    template <typename OutputData>
    auto discover_patternset(const std::string& file, OutputData& out)
    {
        SummaryMiner miner = SummaryMiner{verbose, save_narrow_cast<int>(top_k_queue_size)};
        init_miner(miner);
        miner.read_data(file);
        miner.mine_summary(
            max_items, early_stopping, max_time.count(), ondemand_candidate_generation);
        miner.summary->iterative_scaling();
        export_summary(miner, out);
        return double(miner.trans_count) * double(miner.summary->get_entropy());
    }

    size_t               max_items                     = 100;
    double               min_frequency                 = 0.05;
    size_t               top_k_queue_size              = 0;
    int                  verbose                       = 3;
    std::chrono::seconds max_time                      = std::chrono::hours{5};
    bool                 ondemand_candidate_generation = true;
    bool                 early_stopping                = true;
    MTVModelCost         model_cost                    = MTVModelCost::MDL;

private:
    void init_miner(SummaryMiner& miner) const
    {
        miner.set_use_items(true);
        miner.set_use_rowmargins(false); // !! do not change this to true, unless you also
                                         // update ``import_data''!!
        miner.set_minsup(min_frequency); // miner.minsup is actually the min-frequency!
        miner.set_maxsize(0);            // maximum itemset size threshold [default=0]
        miner.set_maxtreewidth(5);       // maximum number of items per group [default=0]
        miner.set_maxgroupsize(10);      // maximum number of itemsets per group [default=0]
        miner.set_penalty(
            static_cast<int>(model_cost)); // type of penalty [1=BIC, 3=MDL; default=3]
    }

    template <typename OutputData>
    static void
    export_summary(SummaryMiner& miner, OutputData& out, bool remove_singletons = true)
    {
        using pattern_type = typename OutputData::pattern_type;
        sd::disc::itemset<pattern_type> buf;
        // Trie<int> trie;
        sd::Trie<int, bool> trie;
        for (size_t i = 0; i < miner.summary->all_itemsets_size; ++i)
        {
            buf.clear();
            const auto& item = *(miner.summary->all_itemsets[i]);
            if (remove_singletons && item.len == 1)
                continue;
            auto view = slice<const int>(item.itemset, item.len);
            // it is very embarissing, but the MTV miner yield itemsets that are not unique
            // within the summary.
            if (trie.contains(view))
                continue;
            trie.insert(view);

            // if(trie.find(item.itemset, item.len) != trie.notfound) continue;
            // trie.add(item.itemset, item.len);
            for (size_t j = 0; j < view.size(); ++j)
                buf.insert(miner.id2item[item.itemset[j]]);
            auto q = miner.summary->frequencies[i];
            out.insert(q, buf);
        }
    }

    template <typename X>
    static void import_data(const X& data, SummaryMiner& miner)
    {
        size_t dim = 0;
        for (size_t j = 0; j < data.size(); ++j)
        {
            dim = std::max(dim, get_dim(data.point(j)));
        }

        miner.trans_count = data.size();
        miner.absminsup   = miner.minsup * data.size();
        miner.item2id.clear();

        // std::vector<std::vector<int>> transaction_ids(dim);
        std::map<int, std::vector<int>> transaction_ids;
        for (size_t j = 0; j < data.size(); ++j)
        {
            iterate_over(data.point(j), [&](auto i) { transaction_ids[i].push_back(j); });
        }
        assert(transaction_ids.size() > 0 && transaction_ids.size() <= dim);

#ifndef NDEBUG
        std::vector<size_t> support(dim);
        for (size_t j = 0; j < data.size(); ++j)
            iterate_over(data.point(j), [&](auto i) { support[i]++; });

        for (size_t i = 0; i < support.size(); ++i)
        {
            assert(support[i] == 0 || support[i] == transaction_ids[i].size());
        }
#endif

        miner.item_count      = transaction_ids.size(); // dim
        miner.freq_item_count = miner.item_count;
        miner.tidlists        = new TidList*[miner.item_count];
        miner.id2item         = new int[miner.item_count];

        int item_id = 0;
        for (auto& [i, tidset] : transaction_ids)
        {
            if (tidset.empty())
                continue;
            miner.item2id[i]        = item_id;
            miner.id2item[item_id]  = i;
            miner.tidlists[item_id] = new TidList(item_id, tidset);
            ++item_id;
        }

        TidList::prepare(save_narrow_cast<int>(data.size()));
    }
};

template <typename S, typename T>
std::tuple<sd::disc::EncodingLength<T>, std::chrono::milliseconds, size_t, bool>
run_mtv_miner(sd::disc::Dataset<S> const& data,
              size_t                      max_num_patterns,
              size_t                      min_support,
              bool                        bic,
              bool                        early_stopping)
{
    sd::disc::LabeledDataset<T, S> mtv_summary;
    const auto                     min_fr = double(min_support) / data.size();
    sd::disc::MTVMiner             mtvm{max_num_patterns, min_fr};
    mtvm.verbose        = 0;
    mtvm.max_time       = std::chrono::hours(2);
    mtvm.early_stopping = early_stopping;
    mtvm.model_cost     = bic ? sd::disc::MTVModelCost::BIC : MTVModelCost::MDL;

    T    value = 0;
    auto ms    = measure([&] { value = mtvm.discover_patternset(data, mtv_summary); });
    {
        sd::disc::LabeledDataset<T, S> next_summary;
        sd::disc::insert_missing_singletons(data, next_summary);
        for (const auto&& x : mtv_summary)
            next_summary.insert(label(x), point(x));
        mtv_summary = next_summary;
    }
    const bool value_broken = std::isnan(value);
    if (value_broken)
    {
        ms += measure([&] {
            MEDistribution<S, T> mtv2(data.dim, data.size());
            for (const auto& [y, x] : mtv_summary)
                mtv2.insert(y, x);
            estimate_model(mtv2);
            value = disc::encode_data(mtv2, data);
        });
    }
    auto mtv_model_cost = encode_model_sdm(mtv_summary, data.size(), bic);
    auto mtv_cost       = EncodingLength<T>{value, mtv_model_cost};
    return {mtv_cost, ms, mtv_summary.size(), value_broken};
}

} // namespace disc
} // namespace sd