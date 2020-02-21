#pragma once

#include <disc/io/ReadDataset.hxx>

#include <trie/Trie.hxx>
#include <trie/flat_hash_map/flat_hash_map.hpp>

namespace sd::disc
{

struct InputOutput
{
    ska::flat_hash_map<size_t, size_t> to_internal;
    ska::flat_hash_map<size_t, size_t> to_external;
    size_t                             next_index = 0;

    template <typename T, typename S>
    void convert_to_internal(const T& is, S& output)
    {
        output.clear();
        for (const auto& i : is)
        {
            output.insert(convert_to_internal(i));
        }
    }

    size_t unsave_convert_to_internal(size_t index) const
    {
        auto p = to_internal.find(index);
        if (p == to_internal.end())
            throw std::domain_error{"Unexpected Class Label"};
        return p->second;
    }

    size_t convert_to_internal(size_t index)
    {
        auto p = to_internal.find(index);
        if (p == to_internal.end())
        {
            to_internal.emplace(index, next_index);
            to_external.emplace(next_index, index);
            next_index = next_index + 1;
            return next_index - 1;
        }
        return p->second;
    }

    size_t convert_to_external(size_t index) const { return to_external.at(index); }

    template <typename S, typename T>
    void convert_to_external(const S& t, T& out) const
    {
        out.clear();
        iterate_over(t, [&](auto i) { out.insert(convert_to_external(i)); });
    }

    template <typename Data>
    void read_dataset(std::istream& in, Data& data)
    {
        using pattern_type = typename Data::pattern_type;
        itemset<pattern_type> buffer;
        // storage_container<pattern_type> buffer;
        io::read_impl(in, [&](const auto& x) {
            // convert_to_internal(x, buffer);
            buffer.clear();
            for (const auto& i : x)
            {
                auto j = convert_to_internal(i);
                io::assert_index<pattern_type>(j);
                buffer.insert(j);
            }

            data.insert(buffer);
        });
    }

    template <typename Data>
    void read_dataset(const std::string& path, Data& data)
    {
        using pattern_type = typename Data::pattern_type;
        itemset<pattern_type> buffer;
        // storage_container<pattern_type> buffer;
        io::read_impl(path, [&](const auto& x) {
            // convert_to_internal(x, buffer);
            buffer.clear();
            for (const auto& i : x)
            {
                auto j = convert_to_internal(i);
                io::assert_index<pattern_type>(j);
                buffer.insert(j);
            }

            data.insert(buffer);
        });
    }

    template <typename Data>
    void translate_internal(Data& data)
    {
        itemset<typename Data::pattern_type> buf;
        buf.reserve(data.dim);
        for (auto /*ref*/ x : data)
        {
            convert_to_internal(point(x), buf);
            point(x).clear();
            point(x).insert(buf);
        }
    }
    template <typename Data>
    void translate_external(Data& data)
    {
        itemset<typename Data::pattern_type> buf;
        buf.reserve(data.dim);
        for (auto /*ref*/ x : data)
        {
            convert_to_external(point(x), buf);
            point(x).clear();
            point(x).insert(buf);
        }
    }

    template <typename U, typename itemset_data_type>
    auto compare(const U& desc, const itemset_data_type& summary) const
    {
        sd::disc::itemset<disc::tag_sparse>                            buffer;
        sd::Trie<disc::index_type, storage_container<disc::tag_dense>> map;

        size_t i = 0;
        for (const auto& a : desc.assignment)
        {
            for (auto j : a)
            {
                const auto& x = summary.point(j);
                // buffer.assign(x.size(), x);

                // buffer.clear();
                // buffer.insert(x);

                convert_to_external(x, buffer);

                auto& m = map[buffer];
                if (i >= m.length())
                    m.resize(i + 1);
                m.set(i);
                buffer.clear();
            }
            i = i + 1;
        }

        return map;
    }

    // Output:

    template <typename T, typename S>
    void write(std::ostream& out, const T& t, S& buffer, const char delim = ' ') const
    {
        convert_to_external(t, buffer);
        iterate_over(buffer, [&](const auto& i) { out << i << delim; });
    }

    template <typename T, typename S>
    void writeln(std::ostream& out, const T& t, S& buffer, const char delim = ' ') const
    {
        write(out, t, buffer, delim);
        out << '\n';
    }

    template <typename S, typename T, typename U>
    void write(std::ostream& out,
               const T&      support,
               const S&      t,
               U&            buffer,
               const char    delim = ' ') const
    {
        write(out, t, buffer, delim);
        out << '(' << support << ')';
    }

    template <typename S, typename T, typename U>
    void writeln(std::ostream& out,
                 const S&      t,
                 const T&      support,
                 U&            buffer,
                 const char    delim = ' ') const
    {
        write(out, t, support, buffer, delim);
        out << '\n';
    }

    template <typename S, typename T>
    void write_dataset(std::ostream& out, const LabeledDataset<S, T>& data) const
    {
        itemset<T> buffer;
        for (auto&& x : data)
        {
            writeln(out, x, buffer);
        }
    }

    template <typename Data>
    void write_dataset(std::ostream& out, const Data& data) const
    {
        using pattern_type = typename Data::pattern_type;
        itemset<pattern_type> buffer;
        for (auto&& x : data)
        {
            writeln(out, x, buffer);
        }
    }

    static void
    write_dictionary(std::ostream&                                               out,
                     const Trie<disc::index_type, storage_container<tag_dense>>& map,
                     bool ignore_singletons = true)
    {

        out << "[\n";
        map.visit_values([&](const auto& pattern, const auto& cliques) {
            if ((!ignore_singletons || pattern.size() != 1) && pattern.size() > 0)
            {
                out << "([";
                out << front(cliques);
                if (!is_singleton(cliques))
                {
                    bool ignore = true;
                    iterate_over(cliques, [&](size_t i) {
                        if (!ignore)
                            out << "," << i;
                        ignore = false;
                    });
                }
                out << "],[";
                for (size_t i = 0; i < pattern.size() - 1; ++i)
                {
                    out << pattern[i] << ",";
                }
                out << pattern[pattern.size() - 1] << "]),\n";
            }
        });
        out << "]\n";
    }

    template <typename Trait>
    void write_assignment(std::ostream& out, const Composition<Trait>& c) const
    {
        auto d = this->compare(c, c.summary);
        write_dictionary(out, d);
    }

    template <typename Trait>
    void write_summary(std::ostream& out, const Composition<Trait>& c) const
    {
        const auto& s = c.summary;
        write_summary(out, s);
    }

    template <typename T, typename S>
    void write_summary(std::ostream& out, const LabeledDataset<T, S>& s) const
    {
        out << "[\n";
        for (size_t i = 0; i < s.size(); ++i)
        {
            out << "[";
            out << convert_to_external(front(s.point(i)));

            iterate_over(s.point(i), [&, ignore = true](size_t j) mutable {
                if (!ignore)
                {
                    out << ", " << convert_to_external(j);
                }
                ignore = false;
            });
            out << "],\n";
        }
        out << "]\n";
    }

    template <typename Trait>
    void write_decomposition(std::ostream& out, const Composition<Trait>& c) const
    {
        itemset<typename Trait::pattern_type> buffer;
        const auto&                           s = c.data;
        for (const auto& [l, x, _] : s)
        {
            write(out, x, buffer);
            out << " [" << l << "]\n";
        }
    }

    template <typename Trait>
    static void write_frequency(std::ostream& out, const Composition<Trait>& c)
    {
        const auto& q = c.frequency;
        out << "[\n";
        for (size_t i = 0; i < q.extent(0); ++i)
        {
            for (size_t j = 0; j < q.extent(1); ++j)
            {
                out << q(i, j) << ", ";
            }
            out << '\n';
        }
        out << "]\n";
    }

    template <typename Trait>
    void write_composition(std::ostream& out, const Composition<Trait>& c) const
    {
        out << "S = ";
        write_summary(out, c);
        out << "\nQ = ";
        write_frequency(out, c);
        out << "\nA = ";
        write_assignment(out, c);
        out << "\nD = {\n";
        write_decomposition(out, c);
        out << "}\n";
    }

    static std::ostream open_out_file(const std::string& filename)
    {
        std::ofstream outfile(filename, std::fstream::out);
        if (!outfile.good())
        {
            std::cerr << "cannot open file <" << filename << '>' << std::endl;
            std::exit(1);
        }
        std::move(outfile);
    }
};

} // namespace sd::disc
