cmake_minimum_required(VERSION 3.12)

# libraries

    add_library(Widget SHARED)
    target_sources(Widget PRIVATE Widget.cpp)
    target_sources(Widget PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Widget.hpp)

# tests

    add_executable(testA Widget.testA.cpp)
    target_link_libraries(testA Widget)
    add_test(NAME testA COMMAND testA)

    add_executable(testB Widget.testB.cpp)
    target_link_libraries(testB Widget)
    add_test(NAME testB COMMAND testB)

    add_executable(testC Widget.testC.cpp)
    target_link_libraries(testC Widget)
    add_test(NAME testC COMMAND testC)    
