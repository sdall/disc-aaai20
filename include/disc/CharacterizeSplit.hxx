#pragma once

#include "disc/MDL.hxx"
#include <desc/CharacterizeComponents.hxx>
#include <desc/Composition.hxx>
#include <desc/PatternAssignment.hxx>
#include <disc/Encoding.hxx>
#include <disc/Settings.hxx>

#include <utility>

namespace sd
{
namespace disc
{

template <typename Trait, typename T>
void resize_after_split(Composition<Trait>&       c,
                        std::pair<size_t, size_t> index,
                        std::vector<T>&           subset_encodings,
                        const Config&             cfg)
{
    using float_type = typename Trait::float_type;

    c.assignment.resize(c.data.num_components());
    c.models.resize(c.data.num_components(), make_distribution(c, cfg));
    subset_encodings.resize(c.data.num_components(), 0);

    if (c.frequency.size() == 0 || c.data.num_components() == 1 ||
        c.frequency.extent(1) > c.data.num_components())
    {
        compute_frequency_matrix(c);
    }
    else
    {
        ndarray<float_type, 2> q({}, c.summary.size(), c.data.num_components() + 1);
        for (auto i : c.frequency.subscripts()) { q(i) = c.frequency(i); }
        for (size_t i = 0; i < c.frequency.extent(0); ++i)
        {
            q(i, q.extent(1) - 1) = c.frequency(i, c.frequency.extent(1) - 1);
        }
        c.frequency = std::move(q);

        assert(c.frequency.extent(1) >= c.data.num_components());
        assert(c.models.size() > index.second);
        assert(c.assignment.size() > index.second);
        assert(c.frequency.extent(1) > index.second);

        compute_frequency_matrix_column(c.data, c.summary, index.first, c.frequency);
        compute_frequency_matrix_column(c.data, c.summary, index.second, c.frequency);
    }

    assert(check_invariant(c));
}

template <typename Trait, typename T>
void update_encoding(const Composition<Trait>& c,
                     EncodingLength<T>&        l,
                     std::vector<T>&           subset_encodings,
                     size_t                    index)
{
    assert(index < c.models.size());

    l.of_data -= subset_encodings[index];

    subset_encodings[index] = log_likelihood(c.models[index], c.data.subset(index));

    l.of_data += subset_encodings[index];
}

template <typename Trait>
void characterize_split(Composition<Trait>&                         c,
                        std::pair<size_t, size_t>                   index,
                        EncodingLength<typename Trait::float_type>& l,
                        std::vector<typename Trait::float_type>&    subset_encodings,
                        const Config&                               cfg,
                        const bool                                  bic)
{
    resize_after_split(c, index, cfg);

    characterize_one_component(c, index.first, cfg);
    characterize_one_component(c, index.second, cfg);

    update_encoding(c, l, subset_encodings, index.first);
    update_encoding(c, l, subset_encodings, index.second);

    l.of_model = bic ? bic::encode_model_bic(c) : mdl::encode_model_mdl(c);

    assert(check_invariant(c));
}

} // namespace disc
} // namespace sd