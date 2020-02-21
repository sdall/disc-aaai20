#pragma once

#include <disc/storage/Dataset.hxx>
#include <disc/storage/Itemset.hxx>
#if __has_include(<charconv>)
#include <charconv>
#endif
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// #include <stupidcsv/stupid_csv.hxx>
namespace sd::disc::io
{

template <typename Tag>
void assert_index([[maybe_unused]] size_t index)
{
#ifndef USE_LONG_INDEX
    if constexpr (std::is_same_v<Tag, sd::disc::tag_sparse>)
    {
        if (index > std::numeric_limits<sparse_index_type>::max())
        {
            throw std::domain_error{
                "[error] the data you have provided has to many dimensions to be stored in a "
                "uint16_t\n"
                "  note: please recompile the application with -DUSE_LONG_INDEX\n\n"};
        }
    }
#endif
}

index_type to_index(std::string_view str)
{
#if __has_include(<charconv>) && 0
    index_type            result = 0;
    [[maybe_unused]] auto e      = std::from_chars(str.data(), str.data() + str.size(), result);
    assert(e.ec == std::errc());
    return result;
#else
    return std::strtoll(str.data(), nullptr, 10);
#endif
}

template <typename BODY>
void tokenize(std::string_view str, std::string_view delims, BODY&& body)
{
    for (auto first = str.cbegin(), second = str.cbegin(), last = str.cend();
         second != last && first != last;
         first = second + 1)
    {
        second = std::find_first_of(first, last, std::cbegin(delims), std::cend(delims));
        if (first != second)
        {
            body(std::string_view(first, second - first));
        }
    }
}

// template <typename BODY>
// void tokenize(std::string_view str, std::string_view delimiters, BODY&& body)
// {
//     // Skip delimiters at beginning.
//     auto current = str.find_first_not_of(delimiters, 0);
//     // Find first "non-delimiter".
//     auto next = str.find_first_of(delimiters, current);

//     while (std::string::npos != next || std::string::npos != current)
//     {
//         // Found a token, add it to the vector.
//         body(str.substr(current, next - current));
//         // Skip delimiters.  Note the "not_of"
//         current = str.find_first_not_of(delimiters, next);
//         // Find next "non-delimiter"
//         next = str.find_first_of(delimiters, current);
//     }
// }

// template <typename DataType>
// void read2(std::istream& in, DataType& result)
//     ifs.seekg(0, in.end);
//     const auto fileSize = static_cast<size_t>(in.tellg());
//     const auto buffer = make_unique<char[]>(fileSize);
//     vector<size_t> nums;
//     in.seekg(0);
//     in.read(buffer,fileSize);

//     auto str = std::string_view(buffer, static_cast<size_t>(in.gcount()));
//     tokenize(str, '\n', [&](const auto& line) {
//         tokenize(
//     });

//     // return as_from_chars(string_view(buffer, static_cast<size_t>(in.gcount())),
//     std::istream::nos);
// }

template <typename Function>
void read_impl(std::istream& in, Function&& fn)
{
    std::string line;
    line.reserve(1024 * 4);
    // check if istream is readable
    std::getline(in, line);
    in.clear();
    in.seekg(0, std::istream::beg);

    std::vector<disc::index_type> row;
    row.reserve(1024);

    // for (auto line : sd::stupid_csv::csv_reader(in, ' ')) {
    //     for (size_t cell : line) {
    //         assert_index(cell);
    //         row.push_back(cell);
    //     }
    //     result.insert(row);
    //     row.clear();
    // }

    while (std::getline(in, line))
    {
        row.clear();
        tokenize(line, " :;|,", [&](auto&& word) { row.push_back(io::to_index(word)); });
        fn(std::as_const(row));
    }
}

template <typename DataType>
void read(std::istream& in, DataType& result)
{
    read_impl(in, [&](const auto& x) {
        assert_index<typename DataType::pattern_type>(x);
        result.insert(x);
    });
}

template <typename Function>
void read_impl(const std::string& file, Function&& fn)
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

    sd::disc::io::read_impl(infile, std::forward<Function>(fn));
    infile.close();
}

template <typename U, typename V>
auto read_fimi(std::istream& in, LabeledDataset<U, V>& result, bool normalize_counts = true)
{
    // fimi-format: e.g.
    // 1 2 3 (3)
    // 4 5 (5)
    // 1 (15)
    // (25)

    // x y z (support); where x,y,z are items and support = support(x,y,z)
    // in row i, an item j corresponds to the index of a column with value 1, i.e D_{ij} = 1.
    //(support) corresponds to the support of the empty set which is the number of rows in the
    // dataset.

    std::string line;
    std::getline(in, line);
    size_t length = 0;
    in.clear();
    in.seekg(0, std::istream::beg);

    const bool labels_in_file = line.find_first_of("(", 0) != std::string::npos;
    if (!labels_in_file)
    {
        struct no_labels_in_file_exception : std::exception
        {
        };
        throw no_labels_in_file_exception{};
    }

    using index_type = disc::index_type;

    std::vector<index_type> row;
    row.reserve(512);
    index_type value;

    while (!in.eof() && in.good())
    {
        if (in.peek() == ' ' || in.peek() == '\n')
        {
            in.get();
        }
        else if (in.peek() == '(')
        {
            in.get();
            in >> value;
            in.get();
            if (!row.empty())
            {
                assert_index<U>(value);
                result.insert(value, row);
                row.clear();
            }
            else
            {
                length = value;
            }
        }
        else if (in >> value)
        {
            row.push_back(value);
        }
    }
    if (length == 0)
    {
        struct fimi_parser_file_size_not_specified_exception : std::exception
        {
        };
        throw fimi_parser_file_size_not_specified_exception{};
    }

    if (normalize_counts)
    {
        for (auto ref : result)
        {
            label(ref) /= length;
        }
    }

    return length;
}

} // namespace sd::disc::io
