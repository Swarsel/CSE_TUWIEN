# ex13: Copy/Move Construction and Assignment for a Forward List

- **Hand out**: November 16, 9am
- **Hand in**: until November 27, end of day, via your git submission repo `ex13`
- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Overview

- You are provided with a resource owning class `List` which implement a single-(forward)-linked list.
- The `List` implements a minimal subset of the interface of a `std::list`, e.g. minimal construction and access semantics.
- The provided implementation is defective, no user-defined copy/move constructors and copy/move assignment operators are present to handle the resource ownership.
- You need to implement those four missing special member functions to fit the needs of the class.

## Task details: Constructors and Destructors

The important parts of the class (w.r.t. to resource ownership) look likes this:

```cpp
class List {
public:
  struct Node {
    value_type value;        ///< stored value
    Node* next = nullptr;    ///< pointer to next node in list
  };

private:
  Node* _head = nullptr; ///< root element
  size_type _size = 0;   ///< current number of nodes in list

public:
  List();                                     // already implemented 
  List(size_type n, const value_type& init);  // already implemented 
  ~List();                                    // already implemented

  List(const List& other) = default;            // defective
  List(List&& other) = default;                 // defective
  List& operator=(const List& other) = default; // defective
  List& operator=(List&& other) = default;      // defective
};
```

- The implementation of the class is separated into header and source file:
    - [`ex13/List.hpp`](ex13/List.hpp): you should not change anything in this file.
    - [`ex13/List.cpp`](ex13/List.cpp): **your implementation happens solely in this file**.
- Three separate tests are available:
    - [`ex13/List.testA.cpp`](ex13/List.testA.cpp) for the copy constructor
    - [`ex13/List.testB.cpp`](ex13/List.testB.cpp) for the copy assignment operator
    - [`ex13/List.testC.cpp`](ex13/List.testA.cpp) for the move constructor
    - [`ex13/List.testD.cpp`](ex13/List.testB.cpp) for the move assignment operator

## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex13                    # sources
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   ├── List.hpp          # class definition 
│   ├── List.cpp          # member function definitions 
│   ├── List.testA.cpp    # ...
│   ├── List.testB.cpp    # ...
│   └── List.testC.cpp    # ...
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
cd home/of/repo/of/ex13
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```

Sequence for performing the benchmarks:

```shell
cd home/of/repo/of/ex13
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release # generate into 'build' folder (now in release mode)
cmake --build build --target benchmark          # build benchmark executable
./build/benchmark/benchmark                     # run benchmark
```

Example output of the final benchmark:
```shell
           List::List(const List&):9.81e-02s
                List::List(List&&):1.87e-06s
List::operator=(const List& other):1.53e-01s
     List::operator=(List&& other):1.18e-06s
```