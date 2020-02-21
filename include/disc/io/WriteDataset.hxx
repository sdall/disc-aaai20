#pragma once

#include <disc/disc/Composition.hxx>
#include <disc/storage/Dataset.hxx>
#include <disc/storage/Itemset.hxx>

#include <trie/Trie.hxx>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace sd
{
namespace disc
{
namespace io
{

template <typename pattern_type>
void write(std::ostream& out, const pattern_type& line, const char delim = ' ')
{
    iterate_over(line, [&](auto i) { out << i << delim; });
}

template <typename pattern_type>
void writeln(std::ostream& out, const pattern_type& line, const char delim = ' ')
{
    write(out, line, delim);
    out << '\n';
}

template <typename T, typename S>
void writeln(std::ostream& out, T support, const S& line, const char delim = ' ')
{
    write(out, line, delim);
    out << '(' << support << ")\n";
}

template <typename T>
void write(std::ostream& out, const Dataset<T>& data)
{
    for (const auto& d : data)
    {
        writeln(out, point(d));
    }
}

template <typename T, typename S>
void write(std::ostream& out, const sd::disc::LabeledDataset<T, S>& data)
{
    for (const auto& d : data)
    {
        writeln(out, label(d), point(d), ' ');
    }
}

template <typename T, typename S>
void write(std::ostream& out, const LabeledDataset<T, S>& data, size_t len)
{
    write(out, data);
    out << '(' << len << ")\n";
}

template <typename Data_Type>
void read(const std::string& file, Data_Type& in)
{
    std::ifstream infile(file, std::fstream::in);
    if (!infile.good())
    {
        std::cerr << "cannot open file <" << file << '>' << std::endl;
        std::exit(1);
    }
    const size_t bufsize = 256 * 1024;
    char         buf[bufsize];
    infile.rdbuf()->pubsetbuf(buf, bufsize);

    sd::disc::io::read(infile, in);
    infile.close();
}

template <typename Data_Type>
void write(const std::string& file, const Data_Type& data)
{
    std::ofstream outfile(file, std::fstream::out);
    if (!outfile.good())
    {
        std::cerr << "cannot open file <" << file << '>' << std::endl;
        std::exit(1);
    }
    sd::disc::io::write(outfile, data);
    outfile.close();
}

template <typename Data_Type>
void write(const std::string& file, const Data_Type& data, size_t len)
{
    std::ofstream outfile(file, std::fstream::out);
    if (!outfile.good())
    {
        std::cerr << "cannot open file <" << file << '>' << std::endl;
        std::exit(1);
    }
    sd::disc::io::write(outfile, data, len);
    outfile.close();
}

void write_dictionary(std::ostream&                                               out,
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

template <typename U, typename itemset_data_type>
auto compare(const U& desc, const itemset_data_type& summary)
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
            buffer.clear();
            buffer.insert(x);
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

template <typename Trait>
void write_assignment(std::ostream& out, const Composition<Trait>& c)
{
    auto d = compare(c, c.summary);
    write_dictionary(out, d);
}

template <typename Trait>
void write_frequency(std::ostream& out, const Composition<Trait>& c)
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
void write_summary(std::ostream& out, const Composition<Trait>& c)
{
    const auto& s = c.summary;
    out << "[\n";
    for (size_t i = 0; i < s.size(); ++i)
    {
        out << "[";
        out << front(s.point(i));

        iterate_over(s.point(i), [&, ignore = true](size_t j) mutable {
            if (!ignore)
            {
                out << ", " << j;
            }
            ignore = false;
        });
        out << "],\n";
    }
    out << "]\n";
}

template <typename Trait>
void write_decomposition(std::ostream& out, const Composition<Trait>& c)
{
    const auto& s = c.data;
    for (const auto& [l, x, _] : s)
    {
        disc::io::write(out, x, ' ');
        out << " [" << l << "]\n";
    }
}

template <typename Trait>
void write_composition(std::ostream& out, const Composition<Trait>& c)
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

} // namespace io
} // namespace disc
} // namespace sd