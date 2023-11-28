# ex12: Constructing/Destruction of a Resource-Owning Class

- **Hand out**: November 9, 9am
- **Hand in**: until November 20, end of day, via your git submission repo `ex12`
- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Overview

- You are provided with a simple resource owning class `Vector` which holds dynamically allocated memory via a raw pointer (unmanaged ownership).
- The `Vector` implements a minimal subset of the interface of a `std::vector`, e.g. minimal construction and access semantics to its elements which are stored contiguously in memory.
- The provided implementation is defective: no user-defined constructors/destructors are present to handle the resource ownership.
- You need to implement a user-defined destructor, copy constructor, and move constructor.


## Task details: Constructors and Destructors

The important parts of the class (w.r.t. to resource ownership) look likes this:

```cpp
class Vector {
  // ...
private:
  // ...
  value_type* _data = nullptr; ///< current data holder
public:
  // ...
  Vector(const Vector& other) = default;    // defective 
  Vector(Vector&& other)= default;          // defective
  ~Vector()= default;                       // defective
  // ...
}
```

- The implementation of the class is separated into header and source file:
    - [`ex12/Vector.hpp`](ex12/Vector.hpp): you should not change anything in this file.
    - [`ex12/Vector.cpp`](ex12/Vector.cpp): **your implementation happens solely in this file**.
- Three separate tests are available:
    - [`ex12/Vector.testA.cpp`](ex12/Vector.testA.cpp) for the destructor
        - Note: expect this test to fail for the `default` implementation (empty body) if you are running it with the *AddressSanitizer* which detects leaks at runtime; without detection, this would be slient leak at runtime.
    - [`ex12/Vector.testB.cpp`](ex12/Vector.testB.cpp) for the copy constructor
        - Note: expect this test to fail for the default implementation (member-wise copy) as it will result in an aliasing ownership.
    - [`ex12/Vector.testC.cpp`](ex12/Vector.testC.cpp) for the move constructor
        - Note: expect this test to fail for the default implementation (member-wise move) as it will result in an aliasing ownership.

## Project Layout

```
.
├── README.md               # this file
├── CMakeLists.txt          # top-level configuration of the project
├── ex12                    # sources
│   ├── CMakeLists.txt      # configuration w.r.t to sources in this folder
│   ├── Vector.hpp          # class definition 
│   ├── Vector.cpp          # member function definitions 
│   ├── Vector.testA.cpp    # ...
│   ├── Vector.testB.cpp    # ...
│   └── Vector.testC.cpp    # ...
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
cd home/of/repo/of/ex12
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```

Sequence for performing the benchmarks:

```shell
cd home/of/repo/of/ex12
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release # generate into 'build' folder (now in release mode)
cmake --build build --target benchmark          # build benchmark executable
./build/benchmark/benchmark                     # run benchmark
```

Example output of the final benchmark:
```shell
Vector::Vector(const Vector&): 0.31s
Vector::Vector(Vector&&):      1.2e-05s
```