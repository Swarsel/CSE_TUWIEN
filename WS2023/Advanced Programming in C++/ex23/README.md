# ex23: Expressing operations using standard library algorithms

- **Hand out**: December 7, 9am
- **Hand in**: until December 18, end of day, via your git submission repo `ex23`
- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Overview

In this exercise it is your task to re-implement a given set of functions in `ex23/Reference.hpp` (namespace `ìue`) in `ex23/Algorithms.hpp`. You keep the function declarations/signatures identical but change the implementation such that no loops are explicitly used but the functionality is not different to the original function.

## Task Details

The functions you have to re-implement are template functions with a single *type template parameter* `C`. They implement algorithms for container type `C` with the following requirements towards the type `C`:
- access to the elements in the container using the `operator[]`; access is possible at any valid position at any time (this is called *random access*).
- a member function `size()` returning the number of elements.
- a nested type `::value_type`, reflecting the type of the elements stored in the container.

For example, the function template `iue::set_to_value` contains the functionality to set all elements in the container to a certain value `value`:
```cpp
template <typename C> void set_to_value(C& ctr, typename C::value_type value) {
  for (std::size_t i = 0; i < ctr.size(); ++i) {
    ctr[i] = value;
  }
}
```

This function works fine, when the passed container `ctr` is an `std::vector`. However, if `C` would be a `std::list` the expression `ctr[i]` would lead to a compiler error as `std::list` does not provide random access via the overloaded `operator[]`.

It is your task to change the implementation of all functions in `include/algorithms.hpp` to be compatible with at least the following three sequential containers:

- `std::vector`
- `std::list`
- `std::array`

Additionally, your re-implementations are **not allowed** to use any explicit loops, but instead should delegate this looping to functions provided by the standard library:

```cpp
#include <algorithm> // std::fill 
template <typename C> void set_to_value(C& ctr, typename C::value_type value) {
  std::fill(ctr.begin(), ctr.end(), value);
}
```

Similarly, you should change re-implement also all other functions.
For an overview and documentation of all algorithms available in the algorithms library of the C++ standard library, see [https://en.cppreference.com/w/cpp/algorithm]([https://en.cppreference.com/w/cpp/algorithm).
For this exercise however, the following functions should be enough sufficient (in no particular order):
- [std::fill](https://en.cppreference.com/w/cpp/algorithm/fill)
- [std::transform](https://en.cppreference.com/w/cpp/algorithm/transform)
- [std::generate](https://en.cppreference.com/w/cpp/algorithm/generate)
- [std::for_each](https://en.cppreference.com/w/cpp/algorithm/for_each)
- [std::reverse](https://en.cppreference.com/w/cpp/algorithm/reverse)
- [std::count_if](https://en.cppreference.com/w/cpp/algorithm/count_if)
- [std::equal](https://en.cppreference.com/w/cpp/algorithm/equal)
- [std::accumulate](https://en.cppreference.com/w/cpp/algorithm/accumulate)

**NOTE: After you finished your modifications in `include/algorithms.hpp`, no explicit `for`-loops should be present.**

Prepare yourself for a discussion of your implementation and try to identify advantages and disadvantages of using algorithms form the standard library instead of explicit loops.

## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex23                    # sources
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   ├── Reference.hpp       # reference implementations using for loops
│   ├── Algorithms.hpp      # re-implementations w/o explicit loops
│   ├── Algorithms.testA.cpp    # testing functionality 
│   :                           
│   └── Algorithms.testI.cpp    # ...
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
cd home/of/repo/of/ex23
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```

