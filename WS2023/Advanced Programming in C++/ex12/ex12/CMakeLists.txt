cmake_minimum_required(VERSION 3.12)

# libraries

    add_library(Vector SHARED)
    target_sources(Vector PRIVATE Vector.cpp)
    target_sources(Vector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Vector.hpp)

# tests

    add_executable(testA Vector.testA.cpp)
    target_link_libraries(testA Vector)
    add_test(NAME testA COMMAND testA)

    add_executable(testB Vector.testB.cpp)
    target_link_libraries(testB Vector)
    add_test(NAME testB COMMAND testB)

    add_executable(testC Vector.testC.cpp)
    target_link_libraries(testC Vector)
    add_test(NAME testC COMMAND testC)    
