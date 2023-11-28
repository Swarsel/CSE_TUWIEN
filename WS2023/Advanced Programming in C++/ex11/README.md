# ex11: Value/Reference/Move-Semantics
## Hand in: until November 6, 4pm, via your git submission repo `ex11`

- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Overview

- You are provided a simple class `Widget` which holds a `std::vector<double>` as member.
- You are provided three *init* functions, each returning a `Widget` constructed from a single function parameter of type `std::vector<double>`
- The three functions differ in the *value category* of the parameter.
- It is your task to optimize the implementations, i.e. making best use of the *value category* of the parameter when construction the returned `Widget`.

## Task details: Improve the implementation of three initialization functions

- This exercise deals with a `Widget` class which holds a single non-reference member `vec` which is of type `Vector = std::vector<dobule>`:
    ```c++
    using Vector = std::vector<double>

    struct Widget {
      Vector vec;
    };
    ```
- You are provided with the implementation of three initalization functions:
    ```c++
    Widget init(Vector vec);
    Widget init(const Vector& vec);
    Widget init(Vector&& rref);
    ```
- The functions are in different namespaces (`value`, `lref`, and `rref`) so their definitions do not collide.
- Your task is to complete/improve the implementation of these functions (i.e. using `std::move()` where appropriate).
- You implement/improve the three functions **solely** in the *source file* [`ex11/Widget.cpp`](ex11/Widget.cpp).
- Specification details are provided for each function in form of comments in the *header file* [`ex11/Widget.hpp`](ex11/Widget.hpp).
- Each function has an associated test:
    - [`ex11/Widget.testA.cpp`](ex11/Widget.testA.cpp) for `init::value`
    - [`ex11/Widget.testB.cpp`](ex11/Widget.testB.cpp) for `init::lref`
    - [`ex11/Widget.testC.cpp`](ex11/Widget.testC.cpp) for `init::rref`
- The tests detect whether you have found the most efficient solution and will fail if you have not, giving you hints what the problem is (this is only true when compiling in debug mode `CMAKE_BUILD_TYPE=Debug`).


## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex11                    # sources
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   ├── Widget.hpp          # class definition and function declarations (public interface)
│   ├── Widget.cpp          # function definitions (private implementation)
│   ├── Widget.testA.cpp    # source for executable of testA
│   ├── Widget.testB.cpp    # ...
│   └── Widget.testC.cpp    # ...
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
cd home/of/repo/of/ex11
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```
Sequence for performing the benchmarks:
```shell
cd home/of/repo/of/ex11
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release # generate into 'build' folder (now in release mode)
cmake --build build --target benchmark          # build benchmark executable
./build/benchmark/benchmark                     # run benchmark
```

Example output of the benchmark (for the non-improved implementation in the handout):
```shell
      value::init(Vector) passing l-value: 0.29s
      value::init(Vector) passing r-value: 0.15s
lref::init(const Vector&) passing l-value: 0.15s
lref::init(const Vector&) passing r-value: 0.15s
     rref::init(Vector&&) passing r-value: 0.15s
```