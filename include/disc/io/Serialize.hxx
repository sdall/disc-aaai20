#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/disc/DiscoverPatternset.hxx>
#include <disc/interfaces/BoostMultiprecision.hxx>
#include <disc/io/MappedIO.hxx>
#include <disc/storage/Dataset.hxx>

#include <nlohmann/json.hpp>

namespace nlohmann
{
#if defined(HAS_QUADMATH)

template <typename T>
void to_json(nlohmann::json& j, boost::multiprecision::float128 const& x)
{
    j = x.str();
}

#endif

void to_json(nlohmann::json& j, sd::sparse_dynamic_bitset<size_t> const& x) { j = x.container; }

void to_json(nlohmann::json& j, sd::sparse_dynamic_bitset<sd::disc::sparse_index_type> const& x)
{
    j = x.container;
}
} // namespace nlohmann

namespace sd::disc
{

using json = nlohmann::json;

auto make_sparse(sd::ndarray<storage_container<tag_dense>> const& data)
{
    sd::ndarray<storage_container<tag_sparse>> sparse;
    storage_container<tag_sparse>              ext;

    ext.reserve(128);
    sparse.reserve(data.size());

    for (const auto& x : data)
    {
        ext.insert(x);
        sparse.push_back(ext);
        ext.clear();
    }

    return sparse;
}

template <typename T>
json build_mapped_array(std::vector<T> const& data, InputOutput const& io)
{
    using namespace nlohmann;

    auto                          j = json::array();
    storage_container<tag_sparse> ext;
    for (const auto& x : data)
    {
        io.convert_to_external(x, ext);
        j.push_back(ext.container);
    }

    return j;
}

void to_json(json& j, DecompositionSettings const& cfg)
{
    j["alpha"]       = cfg.alpha;
    j["bic"]         = cfg.use_bic;
    j["min_support"] = cfg.min_support;

    j["distribution"]["max_width_per_factor"]    = cfg.distribution.max_width_per_factor;
    j["distribution"]["max_num_sets_per_factor"] = cfg.distribution.max_num_sets_per_factor;

    if (cfg.relaxed_distribution)
    {
        j["relaxed"]["mode"]  = cfg.relaxed_distribution->mode;
        j["relaxed"]["limit"] = cfg.relaxed_distribution->budget_limit;
    }
}

template <typename T>
void to_json(json& j, sd::sparse_dynamic_bitset<T> const& x)
{
    j = x.container;
}

template <typename T>
void to_json(json& builder, EncodingLength<T> const& x)
{
    builder["total"]          = x.objective();
    builder["log_likelihood"] = x.of_data;
    builder["regularizer"]    = x.of_model;
}

template <typename T, typename S>
void to_json(json& builder, LabeledDataset<T, S> const& data)
{
    builder["size"]        = data.size();
    builder["frequencies"] = data.template col<0>().container();

    if constexpr (std::is_same_v<S, tag_sparse>)
    {
        builder["sets"] = data.template col<1>().container();
    }
    else
    {
        builder["sets"] = make_sparse(data.template col<1>()).container();
    }
}

template <typename S>
void to_json(json& builder, PartitionedData<S> const& data)
{
    builder["num_components"] = data.num_components_backup;
    builder["size"]           = data.size();
    builder["component_id"]   = data.template col<0>().container();
    builder["original_index"] = data.template col<2>().container();

    if constexpr (std::is_same_v<S, tag_sparse>)
    {
        builder["sets"] = data.template col<1>().container();
    }
    else
    {
        builder["sets"] = make_sparse(data.template col<1>()).container();
    }
}

template <typename S>
void to_json(json& builder, Dataset<S> const& data)
{
    builder["size"] = data.size();

    if constexpr (std::is_same_v<S, tag_sparse>)
    {
        builder["sets"] = data.template col<0>().container();
    }
    else
    {
        builder["sets"] = make_sparse(data.template col<0>()).container();
    }
}

template <typename T, typename S>
void to_json(json& builder, LabeledDataset<T, S> const& data, const InputOutput& io)
{
    builder["size"]        = data.size();
    builder["dim"]         = data.dim;
    builder["frequencies"] = data.template col<0>().container();
    builder["sets"]        = build_mapped_array(data.template col<1>().container(), io);
}

template <typename S>
void to_json(json& builder, PartitionedData<S> const& data, const InputOutput& io)
{
    builder["num_components"] = data.num_components_backup;
    builder["size"]           = data.size();
    builder["dim"]            = data.dim;
    builder["component_id"]   = data.template col<0>().container();
    builder["original_index"] = data.template col<2>().container();
    builder["sets"]           = build_mapped_array(data.template col<1>().container(), io);
}

template <typename S>
void to_json(json& builder, Dataset<S> const& data, const InputOutput& io)
{
    builder["size"] = data.size();
    builder["dim"]  = data.dim;
    builder["sets"] = build_mapped_array(data.template col<0>().container(), io);
}

template <typename T>
void to_json(json& builder, ndarray<T, 2> const& q)
{
    builder["extents"] = {q.extent(0), q.extent(1)};
    builder["data"]    = q.container();
}

void to_json(json& builder, AssignmentMatrix const& a) { builder["data"] = a; }

template <typename T>
void to_json(json& builder, Composition<T> const& x)
{
    builder["objective"]         = x.encoding;
    builder["initial_objective"] = x.initial_encoding;
    builder["patternset"]        = x.summary;
    builder["frequencies"]       = x.frequency;
    builder["assignment"]        = x.assignment;
    builder["data"]              = x.data;
    builder["num_patterns"]      = x.summary.size() - x.data.dim;
}

template <typename T>
void to_json(json& builder, PatternsetResult<T> const& x)
{
    builder["objective"]         = x.encoding;
    builder["initial_objective"] = x.initial_encoding;
    builder["patternset"]        = x.summary;
    builder["num_patterns"]      = x.summary.size() - x.data.dim;
    builder["data"]              = x.data;
}

template <typename T>
void to_json(json&                 builder,
             Composition<T> const& x,
             InputOutput const&    io,
             bool                  contains_data = true)
{
    builder["objective"]         = x.encoding;
    builder["initial_objective"] = x.initial_encoding;
    builder["frequencies"]       = x.frequency;
    builder["assignment"]        = x.assignment;
    builder["num_patterns"]      = x.summary.size() - x.data.dim;
    to_json(builder["patternset"], x.summary, io);
    if (contains_data)
        to_json(builder["data"], x.data, io);
    else
        builder["data"]["component_id"] = x.data.template col<0>().container();
}

template <typename T>
void to_json(json&                      builder,
             PatternsetResult<T> const& x,
             InputOutput const&         io,
             bool                       contains_data = true)
{
    builder["objective"]         = x.encoding;
    builder["initial_objective"] = x.initial_encoding;
    builder["num_patterns"]      = x.summary.size() - x.data.dim;
    to_json(builder["patternset"], x.summary, io);
    if (contains_data)
        to_json(builder["data"], x.data, io);
}

template <typename Result, typename T>
std::string to_json_string(Result&                               result,
                           std::chrono::milliseconds             elapsed_time,
                           std::vector<EncodingLength<T>> const& progress,
                           InputOutput const&                    io)
{
    // json j = result;
    json j;
    to_json(j, result, io);
    j["elapsed_milliseconds"] = elapsed_time.count();
    j["likelihood_ratio"] = result.encoding.objective() / result.initial_encoding.objective();
    j["progress"]         = progress;
    return j.dump();
}

template <typename Result>
std::string
to_json_string(Result& result, std::chrono::milliseconds elapsed_time, InputOutput const& io)
{
    // json j = result;
    json j;
    to_json(j, result, io);
    j["elapsed_milliseconds"] = elapsed_time.count();
    j["likelihood_ratio"] = result.encoding.objective() / result.initial_encoding.objective();
    return j.dump();
}

template <typename Result>
std::string to_json_string(Result& result, InputOutput const& io)
{
    // json j = result;
    json j;
    to_json(j, result, io);
    j["likelihood_ratio"] = result.encoding.objective() / result.initial_encoding.objective();
    return j.dump();
}

std::vector<char> slurp(const std::string& name)
{
    std::ios_base::sync_with_stdio(false);
    std::ifstream   file(name, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> content(size);

    if (!file.read(content.data(), size))
    {
        throw std::exception();
    }

    return content;
}

auto file_to_json(const std::string& path)
{
    using json   = nlohmann::json;
    auto content = slurp(path);
    return json::parse(content.begin(), content.end());
}

template <typename Data>
void remove_class_labels(Data& data, const std::vector<size_t>& labels)
{
    size_t dim = 0;
    for (auto /* ref */ t : data)
    {
        auto& x = point(t);
        for (auto i : labels)
            x.erase(i);
        dim = std::max(dim, get_dim(x));
    }
    data.dim = dim;
}

template <typename S>
void get_original_class_labels_per_row(const PartitionedData<S>&  data,
                                       const std::vector<size_t>& labels,
                                       std::vector<size_t>&       row_classes)
{
    row_classes.resize(data.size());
    for (auto [y, x, o] : data)
    {
        auto   yo = y;
        size_t c  = 0;
        for (auto i : labels)
        {
            if (is_subset(i, x))
            {
                yo = i;
                c++;
            }
        }
        if (c != 1)
            throw std::domain_error{"not a class label."};
        row_classes[o] = yo;
    }
}

template <typename S>
void split_on_class_labels(PartitionedData<S>&        data,
                           const std::vector<size_t>& labels,
                           std::vector<size_t>&       row_classes)
{
    row_classes.resize(data.size());
    for (auto [y, x, o] : data)
    {
        size_t c = 0;
        for (auto i : labels)
        {
            if (is_subset(i, x))
            {
                y = i;
                c++;
            }
        }
        if (c != 1)
            throw std::domain_error{"not a class label."};
        row_classes[o] = y;
    }
    data.group_by_label();
}

template <typename Data>
nonstd::optional<std::vector<size_t>>
import_data_from_json(nlohmann::json& json, InputOutput& io, Data& data)
{
    using tag = typename Data::pattern_type;
    itemset<tag> row;
    for (const auto& list : json["data"])
    {
        for (size_t i : list)
        {
            auto j = io.convert_to_internal(i);
            row.insert(j);
        }
        data.insert(row);
        row.clear();
    }

    if (data.num_columns() > 1 && json.count("label"))
    {
        using container_t = std::decay_t<decltype(data.template col<0>().container())>;
        using value_type  = typename container_t::value_type;
        data.template col<0>().container() = json["label"].get<std::vector<value_type>>();
    }

    if (json.count("included_class_labels"))
    {
        std::vector<size_t> labels = json["included_class_labels"];
        for (auto& y : labels)
            y = io.convert_to_internal(y);

        std::vector<size_t> row_labels;
        split_on_class_labels(data, labels, row_labels);
        // get_original_class_labels_per_row(data, labels, row_labels);
        remove_class_labels(data, labels);
        return row_labels;
    }

    return {};
}

template <typename Data>
nonstd::optional<std::vector<size_t>>
read_json_data(const std::string& path, InputOutput& io, Data& data)
{
    auto j = file_to_json(path);
    return import_data_from_json(j, io, data);
}

} // namespace sd::disc