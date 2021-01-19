
#include <disc/interfaces/BoostMultiprecision.hxx>

#include <disc/desc/Desc.hxx>
#include <disc/disc/Disc.hxx>

#include <bindings/python/Utilities.hxx>

PYBIND11_MODULE(disc, m)
{
    using namespace pybind11::literals;
    using namespace sd::disc;
    using namespace sd::disc::pyutils;

    m.doc() =
R"doc(DISC: Discover and Describe Data Partitions
-------------------------------------------------
This module contains the python interface to DISC, a method for discovering the pattern composition of a dataset.
The pattern composition consists the following:
    (1) an interpretable partitioning of the data into components in which patterns follow a significantly different distribution
    (2) an description of the partitioning using characteristic and shared patterns
)doc";

    m.def("desc",
          &describe_partitions,
          "Discovers informative patterns using the maxent distribution",
          "dataset"_a,
          "labels"_a.none(true)           = py::list(),
          "min_support"_a                 = 2,
          "is_sparse"_a                   = false,
          "use_higher_precision_floats"_a = false);
    m.def("disc",
          &discover_composition,
          "Discover differently distributed partitions that are characterized using patterns"
          "dataset"_a,
          "min_support"_a                 = 2,
          "alpha"_a                       = 0.05,
          "is_sparse"_a                   = false,
          "use_higher_precision_floats"_a = false);

    m.attr("__version__") = "dev";
}
