cmake_minimum_required(VERSION 3.12)

# header only library

    add_library(Distribution INTERFACE)
    target_sources(Distribution INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Distribution.hpp)       

# tests

    add_executable(testA Distribution.testA.cpp)
    add_test(NAME testA COMMAND testA)

    add_executable(testB Distribution.testB.cpp)
    add_test(NAME testB COMMAND testB)

    add_executable(testC Distribution.testC.cpp)
    add_test(NAME testC COMMAND testC)

    add_executable(testD Distribution.testD.cpp)
    add_test(NAME testD COMMAND testD)

    add_executable(testE Distribution.testE.cpp)
    add_test(NAME testE COMMAND testE)


