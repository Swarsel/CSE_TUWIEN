# ex22: Template class with a templated constructor function

- **Hand out**: November 30, 9am
- **Hand in**: until December 11, end of day, via your git submission repo `ex22`
- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Overview

You task is to implement a template class from scratch which holds the *arithmetic mean*

$
\bar x = \frac{1}{N}\sum_{i=1}^N {x_i}= \frac{x_1 + x_2 + \dots + x_i}{N}
$

and the *standard deviation* 

$
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N {(x_i-\bar x)^2}}
$

originating from a set of samples $x$. 

In a first step your class should be constructible directly from these two values and alternatively by passing the distribution samples in form of a `std::vector`.

In a second step you class needs to be extended to support also other sequential containers to hold the samples passed to the constructor.

## Task 1: `Distribution` class template (construction from a `std::vector`)

The type you implement must be called `Distribution` and be a class template with:

- a single template type, e.g. `template<class T> class {};` or `template<class T> struct {};`
- two public member variables `mean` and `stddev`, both of type `T`
- a public type alias `value_type` which aliases the template parameter `T`
- two constructors:
  - `Distribution(T mean, T stddev)` which directly initializes the two member variables from the parameters (without any calculations).
  - `Distribution(const std::vector<T> &vec)` which calculates the *arithmetic mean* and the *standard deviation* for the distribution of values present in the `std::vector<T>` and initializes the members with these calculated values.

Further requirements on the two constructors describe above are:

1. an implicit deduction mechanism of the class template parameter `T` must be present.
2. the following types for `T` must be supported:
  - `double`
  - `float`
  - `VecN<double, 3>` (from `VecN.hpp`)

**Hints**: The provided `VecN` already supports the required arithmetic operators to calculate the mean and standard deviation including a `sqrt` function defined in `VecN.hpp` (for the built-in types `std::sqrt` is available in `<cmath>`).


**Hint**: When accessing the elements of the passed `std::vector` object, ideally you would use generic access methods from the beginning (i.e. range-base for loops or iterators), so you do not have to change your code for Task 2.

## Task 2: Extension of the `Distribution` class (to support a set of containers for construction)

Extend the available construction mechanism to support the calculation of *arithmetic mean* and *standard deviation* from other containers as well.

At least the following container types must be supported:

- `std::vector`
- `std::list`
- `std::deque`

In order to support these construction mechanisms, your constructor has become a template function itself, and `T` needs to be deduced from the passed container template using the template directly or using a deduction guide.

The template parameter `T` must still be deduced implicitly, i.e. without providing it explicitly when calling the constructors.

**Hint**: For standard library containers, the public type `value_type` allows to access the type of the stored elements.


## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex22                    # sources
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   ├── VecN.hpp            # header only library (template class)
│   ├── Distribution.hpp    # header only library (template class)
│   ├── Distribution.testA.cpp    # testing functionality 
│   ├── Distribution.testB.cpp    # ...
│   ├── Distribution.testC.cpp    # ...
│   ├── Distribution.testD.cpp    # ...
│   └── Distribution.testE.cpp    # ...
├── benchmark               # source of benchmark executable
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   └── main.cpp            # benchmark executable
├── doxygen                 # doxygen configuration
│   ├── CMakeLists.txt      
│   └── doxygen-awesome.css
├── .clang-format           # format style for c++ source code
├── .expected-files         # list of files considered for testing your submission
├── .tests                  # list of tests performed on your submission
├── .gitignore              # file patterns to be ignored by git by default when committing
└── .git                    # internal git bookkeeping
```

# Generate config, build, run, and test using CMake/CTest

Final sequence of commands before submitting:
```shell
cd home/of/repo/of/ex22
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```

Sequence for performing the benchmarks:

```shell
cd home/of/repo/of/ex22
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release # generate into 'build' folder (now in release mode)
cmake --build build --target benchmark          # build benchmark executable
./build/benchmark/benchmark                     # run benchmark
```

Example output of the final benchmark:
```shell
...
```