cmake_minimum_required(VERSION 3.12)

# header only library

    add_library(Wrappers INTERFACE)
    target_sources(Wrappers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Wrappers.hpp)    

# tests

    add_executable(testA Wrappers.testA.cpp)
    add_test(NAME testA COMMAND testA)

    add_executable(testB Wrappers.testB.cpp)
    add_test(NAME testB COMMAND testB)

    add_executable(testC Wrappers.testC.cpp)
    add_test(NAME testC COMMAND testC)
