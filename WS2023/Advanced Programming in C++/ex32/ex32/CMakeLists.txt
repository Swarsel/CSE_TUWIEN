cmake_minimum_required(VERSION 3.12)

# header only library

    add_library(unique_ptr INTERFACE)
    target_sources(unique_ptr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/unique_ptr.hpp)       

# tests

    add_executable(testA unique_ptr.testA.cpp)
    add_test(NAME testA COMMAND testA)

    add_executable(testB unique_ptr.testB.cpp)
    add_test(NAME testB COMMAND testB)

    add_executable(testC unique_ptr.testC.cpp)
    add_test(NAME testC COMMAND testC)

    add_executable(testD unique_ptr.testD.cpp)
    add_test(NAME testD COMMAND testD)




