#pragma once

namespace sd::disc
{

template <typename S>
double purity(const PartitionedData<S>&  data,
              const std::vector<size_t>& classes,
              const std::vector<size_t>& class_labels)
{
    assert(classes.size() == data.size());

    double              aggregated_purity = 0;
    std::vector<size_t> h(class_labels.size());
    for (size_t k = 0; k < data.num_components(); ++k)
    {
        auto component = data.subset(k);
        for (size_t i = 0; i < component.size(); ++i)
        {
            size_t original_position = get<2>(component[i]);
            auto   p =
                std::find(begin(class_labels), end(class_labels), classes[original_position]);
            ++h[std::distance(begin(class_labels), p)];
        }
        auto m      = *std::max_element(begin(h), end(h));
        auto purity = double(m); // / component.size();
        aggregated_purity += purity;
        std::fill(begin(h), end(h), 0);
    }
    return aggregated_purity / data.size();
    // return aggregated_purity / data.num_components();
}

template <typename S>
double purity(const PartitionedData<S>& data, const std::vector<size_t>& classes)
{
    assert(classes.size() == data.size());

    auto d = *std::max_element(classes.begin(), classes.end());

    double              aggregated_purity = 0;
    std::vector<size_t> h(d + 1);
    for (size_t k = 0; k < data.num_components(); ++k)
    {
        auto component = data.subset(k);
        for (size_t i = 0; i < component.size(); ++i)
        {
            size_t original_position = get<2>(component[i]);
            ++h[classes[original_position]];
        }
        aggregated_purity += *std::max_element(begin(h), end(h));
        std::fill(begin(h), end(h), 0);
    }
    return aggregated_purity / data.size();
    // return aggregated_purity / data.num_components();
}

} // namespace sd::disc