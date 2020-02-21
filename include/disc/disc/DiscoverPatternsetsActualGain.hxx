#pragma once

#include <disc/disc/CharacterizeComponents.hxx>
#include <disc/disc/Composition.hxx>
#include <disc/disc/Encoding.hxx> // AssignmentMatrix, encoding(..)
#include <disc/disc/Frequencies.hxx>
#include <disc/disc/SlimCandidateGeneration.hxx>
#include <disc/distribution/Distribution.hxx> // mtv-model
#include <disc/interfaces/BoostMultiprecision.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/utilities/EmptyCallback.hxx>
#include <disc/utilities/PvalueNHC.hxx>
#include <disc/utilities/Support.hxx> // frequency, support
#include <disc/utilities/WeakNumberComposition.hxx>

#include <chrono>
#include <cmath>
#include <nonstd/optional.hpp>

namespace sd
{
namespace disc
{

template <typename Trait, typename Pattern>
std::pair<bool, Composition<Trait>>
find_assignment_actual_gain(Composition<Trait> c,
                            size_t             candidate_index,
                            const Pattern&     candidate_itemset,
                            bool               use_bic,
                            const double       alpha = 0.05)
{

    using float_type = typename Trait::float_type;

    size_t has_assigned = 0;
#pragma omp parallel for reduction(+ : has_assigned)
    for (size_t i = 0; i < c.assignment.size(); ++i)
    {
        const auto& x      = candidate_itemset;
        const auto  subset = c.data.subset(i);
        const auto  q      = frequency<float_type>(x, subset);

        if (!c.models[i].is_item_allowed(x))
        {
            continue;
        }

        auto model = c.models[i];
        model.insert(q, x);
        estimate_model(model);

        auto next = disc::encode_data(model, subset);
        if (std::isnan(next) || std::isinf(next) || next < 0)
            continue;
        if (disc::nhc_pvalue(c.subset_encodings[i], next) < float_type(alpha))
        {
            c.subset_encodings[i] = next;
            c.assignment[i].insert(candidate_index);
            c.models[i] = std::move(model);
            ++has_assigned;
        }
    }

    if (has_assigned > 0)
    {
        c.encoding = encoding_length_mdm(c, use_bic);
    }

    return {has_assigned > 0, c};
}

template <typename Trait, typename Pattern>
std::pair<bool, Composition<Trait>> find_assignment_actual_gain(
    Composition<Trait> c, const Pattern& candidate_itemset, const double alpha = 0.05)
{
    const auto candidate_index = c.summary.size();
    auto       r =
        find_assignment_actual_gain(std::move(c), candidate_index, candidate_itemset, alpha);
    if (r.first)
    {
        auto g = frequency<typename Trait::float_type>(candidate_itemset, r.second.data);
        insert_pattern_to_summary(r.second, candidate_itemset, g);
    }
    return r;
}

template <typename Trait, typename pattern_t, typename V>
void insert_pattern_to_summary(Composition<Trait>& c,
                               const pattern_t&    pattern,
                               const V&            glob_frequency)
{
    assert(c.frequency.extent(1) > 0);

    c.summary.insert(glob_frequency, pattern);

    c.frequency.push_back();
    auto new_q = c.frequency[c.frequency.length() - 1];

    for (size_t j = 0; j < c.data.num_components(); ++j)
    {
        new_q(j) = frequency<typename Trait::float_type>(pattern, c.data.subset(j));
    }
    new_q.back() = glob_frequency;
}

template <typename T>
void initialize_or_update(Composition<T>& c, MiningSettings cfg)

{
    if (check_invariant(c))
    {
        initialize_composition(c, cfg);
    }
    else
    {
        retrain_models(c);
    }

    c.initial_encoding = c.encoding;
    c.encoding         = encoding_length_mdm(c, cfg.use_bic);
}

template <typename Trait, typename Generator, typename CALLBACK = EmptyCallback>
Composition<Trait> discover_patternsets_test_actual_gain(Composition<Trait> comp,
                                                         Generator&         generator,
                                                         MiningSettings     cfg      = {},
                                                         CALLBACK&&         callback = {})
{
    using float_type = typename Trait::float_type;
    // using value_type = T;

    initialize_or_update(comp, cfg);

    assert(comp.data.num_components() > 0);
    assert(comp.data.dim != 0);

    auto& models = comp.models;

    // std::vector<T> encodings = make_data_encodings<S, T>(comp, models);
    // comp.subset_encodings = make_data_encodings<S, T>(comp);
    comp.encoding         = encoding_length_mdm(comp, cfg.use_bic);
    comp.initial_encoding = comp.encoding;
    assert(!std::isnan(comp.encoding.objective()));

    if (cfg.max_iteration == 0)
    {
        return comp;
    }

    constexpr size_t max_patience = 3;
    size_t           patience     = max_patience;
    size_t           items_used   = 0;

    auto score_function = [&](const auto& x, const auto&...) {
        auto ps = comp.models; // very expensive!
        for (size_t i = 0; i < ps.size(); ++i)
        {
            const auto& d = comp.data.subset(i);
            ps[i].insert(disc::frequency<float_type>(x.pattern, d), x.pattern);
            estimate_model(ps[i]);
        }
        return encode_subsets(ps, comp.data);
        // return mean_score(comp, x, cfg.use_bic);
    };
    auto get_allowance = [&models](const auto& t) {
        return std::any_of(
            begin(models), end(models), [&](auto& m) { return m.is_item_allowed(t); });
    };

    generator.compute_scores(score_function);
    generator.prune([&](const auto& t) { return t.score <= 0 || !get_allowance(t.pattern); });
    generator.order_candidates();

    callback(comp.encoding.objective(), std::numeric_limits<float_type>::infinity(), 0);

    const auto start_time = std::chrono::system_clock::now();
    for (size_t it = 0; it < cfg.max_iteration; ++it)
    {
        if (cfg.max_patternset_size && items_used > cfg.max_patternset_size.value())
            break;
        if (cfg.max_time && std::chrono::system_clock::now() - start_time > *cfg.max_time)
            break;

        if (!generator.has_next())
            break;

        auto c_opt = generator.next();
        if (!c_opt)
            continue;

        auto c = std::move(c_opt.value());

        auto [has_assigned, next] =
            find_assignment_actual_gain(std::move(comp), c.pattern, cfg.alpha);

        comp = std::move(next);

        if (has_assigned)
        {
            callback(std::as_const(comp));

            patience   = std::max(max_patience, size_t(patience * 2));
            items_used = items_used + 1;

            generator.add_next(c, score_function);
            generator.prune(
                [&](const auto& t) { return t.score <= 0 || !get_allowance(t.pattern); });
        }
        else
        {
            if (patience-- == 0)
                break;
        }
    }

    // comp.subset_encodings = std::move(encodings);

    return comp;
}
template <typename Trait, typename CALLBACK = EmptyCallback>
Composition<Trait> discover_patternsets_test_actual_gain(Composition<Trait>    com,
                                                         const MiningSettings& cfg      = {},
                                                         CALLBACK&&            callback = {})

{
    using float_type   = typename Trait::float_type;
    using pattern_type = typename Trait::pattern_type;
    auto gen           = SlimGenerator<pattern_type, float_type>{
        com.data, cfg.min_support, cfg.max_pattern_size};
    return discover_patternsets_test_actual_gain(std::move(com), gen, cfg, callback);
}

} // namespace disc
} // namespace sd