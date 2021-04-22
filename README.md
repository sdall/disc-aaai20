# DISC - Explainable Data Decompositions (AAAI-20)

This project contains our implementation of `DISC`, a method for discovering and explaining components in binary datasets. For more details see [S. Dalleiger and J. Vreeken, Explainable Data Decompositions, AAAI (2020)](http://eda.mmci.uni-saarland.de/disc/)

## Abstract

Our goal is to partition a dataset into components, but unlike regular clustering algorithms we additionally want to describe _why_ there are components, _how_ they are different from each other, and what properties are _shared_ among components, in terms of informative patterns observed in the dataset.
Our notion of components is therefore different from the typical definition of clusters, which are often defined as a high-similarity or high-density regions in the dataset. Instead, we consider components as regions in the data that show significantly different _pattern_ distributions. 

We define the problem in terms of a regularized maximum likelihood, in which we use the Maximum Entropy principle to model each data component with a set of patterns. As the search space is large and unstructured, we propose the deterministic DISC algorithm to efficiently discover high-quality decompositions via an alternating optimization approach. Empirical evaluation on synthetic and real-world data shows that DISC efficiently discovers meaningful components and accurately characterises these in easily understandable terms. 

## Dependencies

1. C++17 compiler, OpenMP
2. Boost
3. TBB 2020.2 (Optional)

**At the moment of this writing, the Parallel STL of g++ (>=10.1.1) requires TBB and is incompatible with [TBB 2021.2](https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html)** 


On Debian or Ubuntu you can obtain these for example using 

```sh
apt install libboost-dev libtbb2 libtbb-dev g++ 
```

On Fedora you can obtain these for example using 

```sh
dnf install boost-devel tbb-devel g++
```

On MacOS you can obtain these for example using Homebrew and

```sh
brew install tbb boost gcc libomp
```

For higher precision floating points we can optionally make use of the non-standard 128 bit float type, however, for this we require GNU Quadmath, Boost.Multiprecision and g++.

## Use DISC and DESC from Python

If the dependencies above are met, you can simply install the python version using
```sh
pip install git+https://github.com/sdall/disc-aaai20.git
```
and simply import the algorithms using 
```python
from disc import disc, desc
```
These functions either expect a binary numpy matrix or a sparse representation of the binary data matrix, that is $B_{ij} = 1$ implies that in row $i$ we can find the value $j$. For example $B = \begin{matrix} 0 & 1 \\ 1 & 1 \end{matrix}$ implies 
```python
data = [[0], [0, 1]]
```
For this given, we can now discover an informative pattern set

```python
desc(data)
```
or we can use a numpy matrix directly, e.g.

```python
import numpy
desc(numpy.random.uniform(size=(10,5)) > 0.5)
```
It returns a python dictionary with information such as the summary, (initial) objective function and frequencies of singletons and patterns. If we have multiple datasets given, i.e. classes, DESC can describe these in terms of characteristic and shared patterns

```python
class_labels = [0, 1]
desc(data, class_labels)
```
Additionally, we are now provided with the assignment matrix that assigns a pattern to component (class) if that pattern is informative for that class.

However, if we have no class labels and if we are interested in the parts of the data that exhibit a significantly different distribution from the rest, we can make use of DISC to jointly discover and describe these, i.e.

```python
disc(data)
```
Again, we are provided with the assignment matrix, but in addition, we also have access to the estimated class labels.

For a complete example see the notebook ```jupyter/disc-iris.ipynb```.

## Use DISC and DESC from R

**The R interface has not received major updates and is unmaintained at the moment**

If all requirements are met, starting from the root-directory of this project you can compile and import the Rcpp based bindings by using

```R
install.packages('Rcpp')
require(Rcpp)
inc = paste('-std=c++17 -DWITH_EXECUTION_POLICIES=1 ', '-I ', getwd(), '/include', ' -I ', getwd(), '/src', sep='')
Sys.setenv("PKG_CXXFLAGS"=inc)
Sys.setenv("PKG_LIBS"="-ltbb") 
sourceCpp("src/bindings/R/RDisc.cpp")
```
Similar to the python interface you now have access to desc and disc

```R
    data   <- list(c(1, 2, 3), c(2, 3, 4))
    labels <- list(0, 1)

    desc(data)
    desc(data, labels)
    disc(data)
```
