# DISC - Explainable Data Decompositions (AAAI-20)

This project contains our implementation of `DISC`, a method for discovering and explaining components in binary datasets. For more details see [S. Dalleiger and J. Vreeken, Explainable Data Decompositions, AAAI (2020)](https://aaai.org/Papers/AAAI/2020GB/AAAI-DalleigerS.8171.pdf)


## Abstract

Our goal is to partition a dataset into components, but unlike regular clustering algorithms we additionally want to describe _why_ there are components, _how_ they are different from each other, and what properties are _shared_ among components, in terms of informative patterns observed in the dataset.
Our notion of components is therefore different from the typical definition of clusters, which are often defined as a high-similarity or high-density regions in the dataset. Instead, we consider components as regions in the data that show significantly different _pattern_ distributions. 

We define the problem in terms of a regularized maximum likelihood, in which we use the Maximum Entropy principle to model each data component with a set of patterns. As the search space is large and unstructured, we propose the deterministic DISC algorithm to efficiently discover high-quality decompositions via an alternating optimization approach. Empirical evaluation on synthetic and real-world data shows that DISC efficiently discovers meaningful components and accurately characterises these in easily understandable terms. 

## Required Dependencies

1. C++17 compiler
2. Boost.Math

The following ```python``` and ```R``` libraries are thin layers around a common ```C++17``` core. 

## Optional Dependencies

3. GNU Quadmath
4. Boost.Multiprecision (for higher-precision float128)
5. pybind11, numpy
6. Rcpp

## Python

```{sh}
    pip install git+https://github.com/sdall/disc-aaai20.git
```

For a complete example see the notebook ```jupyter/disc-iris.ipynb```.

```{python}
    import disc
    data = [
        [1, 2, 3], 
        [1, 4, 5], 
        ...
    ]

    # DESC: Discover patternset for a single dataset

    c = disc.discover_patternset(data)
    for a,b in zip(["Set", "Frequencies", "Initial BIC", "BIC"], c):
        print(str(a) + ' ' + str(b))

    # DESC: Discover pattern-composition for decomposed dataset
    #       The labels correspond to the component to which a row in the dataset belongs to 

    labels=[0, 1, ...]
    c = disc.characterize_partitions(data, labels)

    # DISC: Jointly decompose the dataset and discover the pattern-composition
    c = disc.discover_composition(data)
```

You can also use our implementation of the ```Maximum Entropy Distribution``` independently from the rest

```{python}
    # make sure that any element i < dim_of_data
    dim_of_data = 10
    Distribution p(dim_of_data)

    p.infer([1,2])
    p.insert(0.1, [1,2])
    p.infer([1,2])
```

## R

```{R}
    require(Rcpp)
    Sys.setenv("PKG_CXXFLAGS"="-I./thirdparty -I./include -I./src")
    sourceCpp("./src/Rcpp/RcppDisc.cpp")
```

```{R}
    library(discminer)
    data = list(c(1, 2, 3), c(1, 4, 5))
    labels <- list(0, 1)

    s <- discover_patternset(data, 0.05, 1)
    s <- characterize_partitions(data, labels, 0.05, 1)
    s <- discover_composition(data, 0.05, 1)

    dim=6
    p <- new(MEDistribution, dim)
    p$infer(c(1, 2))
    p$insert(0.1, c(1, 2))
    p$infer(c(1, 2))
```
