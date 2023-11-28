# ex21: Wrapper function: track execution time of arbitrary function calls

- **Hand out**: November 23, 9am
- **Hand in**: until December 4, end of day, via your git submission repo `ex21`
- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Overview

It is your task to generialize/adopt a provided wrapper-functionality to measure the run time of *callables*, which can be functions or function objects which can be *called* using the `()` operator.

## Task details

The provided starting point looks is provided in form of the template function `ex21::timeit` in `ex21/Wrappers.hpp` which only works for wrapping function signatures with exactly two copy-constructable parameters of the **same type**:

```cpp

struct AutoTimer { ... };

template <typename CALLABLE, typename ARG>
decltype(auto) timeit(CALLABLE&& callable, ARG arg1, ARG arg2) {
  auto timer = AutoTimer();
  return callable(arg1, arg2);
}
```

However, the provided implementation will fail
- if two parameters with different type are passed to `ex21::timeit`, or
- if the number of parameters is not exactly 2.

Be aware that your solution must support any number of parameters and therefore adding overloads with different numbers of arguments will not lead you to a reasonable solution.

In order to allow for any number of different parameters, you should use

- **perfect forwarding** to support any combination of "value-ness", "const-ness", and "reference-ness" for the list of arguments in combination with a
- **variadic template (template parameter pack)** to be able to forward any number of arguments.

You have to implement the wrapping-functionality solely in `ex21/Wrappers.hpp` and are not allowed to change any other files.

## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex21                    # sources
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   ├── Wrappers.hpp          # header only library (template functions)
│   ├── Wrappers.testA.cpp    # testing functionality 
│   ├── Wrappers.testB.cpp    # ...
│   └── Wrappers.testC.cpp    # ...
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
cd home/of/repo/of/ex21
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```

Sequence for performing the benchmarks:

```shell
cd home/of/repo/of/ex21
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release # generate into 'build' folder (now in release mode)
cmake --build build --target benchmark          # build benchmark executable
./build/benchmark/benchmark                     # run benchmark
```

Example output of the final benchmark:
```shell
push_back(const Vector&)            : 2.59e+01ms
emplace_back(/*ctor args*/);        : 1.94e+01ms
push_back(Vector&&); from std::move : 1.21e-03ms
push_back(Vector&&); from temporary : 1.16e-03ms

```