# ex0: Some helper functions for `std::vector<double>`
## Hand in: until October 16, 4pm, via your git submission repo `ex0`

- This is an introductory exercise.
- It aims to help you to get started with your own development environment.
- The exercise is not graded, nevertheless we **require** a submission to make you comfortable with the modalities for the later exercises.
- It is **not** a problem if you do not complete all required function implementations.

Requirements and recommendations towards your development environment: [360251/setup](https://sgit.iue.tuwien.ac.at/360251/setup)

Details on the git-based hand-out/hand-in/feedback workflow: [360251/git](https://sgit.iue.tuwien.ac.at/360251/git)

Details on the utilization of the provided CMake configuration: [360251/cmake](https://sgit.iue.tuwien.ac.at/360251/cmake)

## Overview

- You are provided a set of function specifications.
- It is your task to implement these functions according to the described specification.
- All functions are bundled into a library with the name `helpers`.
- A single test is already provided for each function.
- Each test links to this library and performs some checks on one of the functions.
- The helper functions all deal with a type `Vector` defined by this alias `using Vector = std::vector<double>`

## Task details: Implement the following seven helper functions

- You implement the functions **solely** in the *source file* [`ex0/helpers.cpp`](ex0/helpers.cpp).
- Specification details are provided for each function in form of comments in the *header file* [`ex0/helpers.hpp`](ex0/helpers.hpp).
- You can also inspect the sources of the affiliated tests (e.g. [`ex0/helpers.testC.cpp`](ex0/helpers.testA.cpp) to infer the required functionality.

```cpp
using Vector = std::vector<T>;
using Compare = std::function<bool(const double& a, const double& b)>;

void print(const Vector& vec);
void reset(Vector& vec);
Vector copy(const Vector& vec);
Vector concat(const Vector& a, const Vector& b);
void swap(Vector& a, Vector& b);
void fill_uniform_random(Vector& vec, std::size_t n, double lower, double upper);
void sort(Vector& vec, Compare comp);
```

## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex0                     # sources
│   ├── CMakeLists.txt          # configuration w.r.t to sources in this folder
│   ├── helpers.hpp             # function declarations (public interface)
│   ├── helpers.cpp             # function definitions (private implementation)
│   ├── helpers.testA.cpp       # source for executable of testA
│   ├── helpers.testB.cpp       # ...
│   ├── helpers.testC.cpp       # ...
│   ├── helpers.testD.cpp       # ...
│   ├── helpers.testE.cpp       # ...
│   ├── helpers.testF.cpp       # ...
│   └── helpers.testG.cpp       # ...
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
cd home/of/repo/of/ex0
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```
