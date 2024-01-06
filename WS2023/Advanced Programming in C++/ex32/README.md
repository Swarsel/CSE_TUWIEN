# ex32: Unique ownership (simplified `std::unique_ptr`)

- **Hand out**: December 21, 9am
- **Hand in**: until January 15, 2024, end of day, via your git submission repo `ex32`
- Your solution to this exercise is the starting point in the 30min discussion session.
- Prepare yourself for a discussion on your solution including the run times you encountered when benchmarking your code.

## Task1 Overview: Implement a simplified `std::unique_ptr`

In this exercise it is your task to implement a *smart pointer* class in `ex32/unique_ptr.hpp`. 
Your implementation `unique_ptr` will be a simplified version of `std::unique_ptr` [(cppref)](https://en.cppreference.com/w/cpp/memory/unique_ptr):

Your class will not support a *deleter deduction* for array types automatically (you will add a *custom deleter function* later in Task2). This allows implicit type deduction on construction for your type (which is not available for `std::unique_ptr` due to the fact that the returned type from a `new T` is indistinguishable from the returned type from `new T[]`).

### Task1 Details: Specification and Tests

Your implementation should provide the following functionality and effects on the public interface:

- **testA** checks: 
  - nested types `::pointer` and `::element_type` reflecting the type of the managed resource
  - `get()` to return a raw pointer to the managed resource
  - construction from a raw pointer
  - move construction from other `unique_ptr`
  - move assignment from other `unique_ptr`
  - disabled copy-construction and copy-assign
- **testB** checks:
  - `reset(arg)` to delete the currently managed resource and start managing the resource passed via a pointer (`arg`)
  - `reset()` to delete the currently managed resource (i.e., behave the same as `reset(nullptr)`)
  - assignment from `nullptr`, which should have the same behavior as `reset()`  
  - `release()` to stop management of the currently wrapped resource and return a pointer to the resource
- **testC** checks:
  - comparison operators `==` and `!=` between two `unique_ptr`s of the same type (should behave like the equivalent raw pointer comparison)
  - comparison operators `==` and `!=` between a `unique_ptr` and a `nullptr` (should behave like a raw pointer comparison with `nullptr`) 
  - implicit conversion to `bool`, i.e. to be used in a condition like this: `if (ptr) {...}` 


## Task2 Overview: Custom deleter function

After implementing all of the functionality above, you are required to add support for a *custom deleter* via a function pointer with this signature: `void (*)(T*)`.

You should introduce an additional class member to hold this function pointer and provide an additional constructor with a second parameter to allow to provide a *custom deleter function*.

If no custom deleter function is supplied (i.e. tests A-C), your implementation should default to calling `delete` on the managed resource.

This custom deleter now allows to handle custom deletion tasks, for example, you can now manage a raw array with your class if you provide the fitting `delete[]` operation via a custom deleter. 

**Hint**: Try to make all previous tests A-C work again once you implemented the default deleter before moving TestD.

### Task1 Details: Specification and Tests

Your implementation of a custom deleter should provide the following functionality:

- **testD** checks: 
  - construction using a custom deleter function pointer as second parameter
  - invocation of custom deleters on destruction of resource
  - a use case where a raw array (to be deleted via `delete[]`) is wrapped using `unique_ptr`
  - a use case where `FILE` handles are wrapped using `unique_ptr` and a custom deleter function is used to 'close' the handle on the ´FILE´
  - considering move assignments (between instances with different custom deleters)


## Project Layout

```
.
├── README.md                # this file
├── CMakeLists.txt           # top-level configuration of the project
├── ex32                     # sources
│   ├── CMakeLists.txt       # configuration w.r.t to sources in this folder
│   ├── unique_ptr.hpp       # doubly-linked list with iters
│   ├── unique_ptr.testA.cpp # testing functionality 
│   :                           
│   └── unique_ptr.testD.cpp # ...
├── doxygen                  # doxygen configuration
│   ├── CMakeLists.txt      
│   └── doxygen-awesome.css
├── .clang-format            # format style for c++ source code
├── .expected-files          # list of files considered for testing your submission
├── .tests                   # list of tests performed on your submission
├── .gitignore               # file patterns to be ignored by git by default when committing
└── .git                     # internal git bookkeeping
```

# Generate config, build, run, and test using CMake/CTest

Final sequence of commands before submitting:
```shell
cd home/of/repo/of/ex32
rm -rf build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug   # generate into 'build' folder
cmake --build build --target all                # build all targets
ctest --test-dir build                          # run all tests
```

