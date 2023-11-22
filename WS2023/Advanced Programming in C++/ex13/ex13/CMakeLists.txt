cmake_minimum_required(VERSION 3.12)

# libraries

    add_library(List SHARED)
    target_sources(List PRIVATE List.cpp)
    target_sources(List PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/List.hpp)

# tests

    add_executable(testA List.testA.cpp)
    target_link_libraries(testA List)
    add_test(NAME testA COMMAND testA)

    add_executable(testB List.testB.cpp)
    target_link_libraries(testB List)
    add_test(NAME testB COMMAND testB)

    add_executable(testC List.testC.cpp)
    target_link_libraries(testC List)
    add_test(NAME testC COMMAND testC)    

    add_executable(testD List.testD.cpp)
    target_link_libraries(testD List)
    add_test(NAME testD COMMAND testD)      