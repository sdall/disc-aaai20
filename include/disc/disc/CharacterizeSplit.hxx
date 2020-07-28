#pragma once

#include <disc/desc/CharacterizeComponents.hxx>
#include <disc/desc/Composition.hxx>
#include <disc/desc/Encoding.hxx>
#include <disc/desc/PatternAssignment.hxx>
#include <disc/disc/Settings.hxx>

#include <utility>

namespace sd
{
namespace disc
{

template <typename Trait>
void resize_after_split(Composition<Trait>&       c,
                        std::pair<size_t, size_t> index,
                        const Config&             cfg)
{
    using float_type = typename Trait::float_type;

    c.assignment.resize(c.data.num_components());
    c.models.resize(c.data.num_components(), make_distribution(c, cfg));
    c.subset_encodings.resize(c.data.num_components(), 0);

    if (c.frequency.size() == 0 || c.data.num_components() == 1 ||
        c.frequency.extent(1) > c.data.num_components())
    {
        compute_frequency_matrix(c.data, c.summary, c.frequency);
    }
    else
    {
        ndarray<float_type, 2> q({}, c.summary.size(), c.data.num_components() + 1);
        for (auto i : c.frequency.subscripts())
        {
            q(i) = c.frequency(i);
        }
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
auto update_encoding(Composition<Trait>& c, std::pair<size_t, size_t> index, const bool bic)
{
    update_encoding(c, c.encoding, c.subset_encodings, index.first);
    update_encoding(c, c.encoding, c.subset_encodings, index.second);

    c.encoding.of_model = encode_model(c, bic);

    return c.encoding;
}

template <typename Trait>
void characterize_split(Composition<Trait>&       c,
                        std::pair<size_t, size_t> index,
                        const Config&             cfg)
{
    resize_after_split(c, index, cfg);

    characterize_one_component(c, index.first, cfg);
    characterize_one_component(c, index.second, cfg);

    assert(check_invariant(c));

    c.encoding = update_encoding(c, index, cfg.use_bic);
}

} // namespace disc
} // namespace sd