cmake_minimum_required(VERSION 3.12)

# libraries

    add_library(helpers SHARED)
    target_sources(helpers PRIVATE helpers.cpp)
    target_sources(helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/helpers.hpp)
    target_include_directories(helpers PUBLIC .)

# tests

    add_executable(testA helpers.testA.cpp)
    target_link_libraries(testA helpers)
    add_test(NAME testA COMMAND testA)
    set_tests_properties(testA PROPERTIES PASS_REGULAR_EXPRESSION "42" "24" "11")

    add_executable(testB helpers.testB.cpp)
    target_link_libraries(testB helpers)
    add_test(NAME testB COMMAND testB)

    add_executable(testC helpers.testC.cpp)
    target_link_libraries(testC helpers)
    add_test(NAME testC COMMAND testC)

    add_executable(testD helpers.testD.cpp)
    target_link_libraries(testD helpers)
    add_test(NAME testD COMMAND testD)

    add_executable(testE helpers.testE.cpp)
    target_link_libraries(testE helpers)
    add_test(NAME testE COMMAND testE)

    add_executable(testF helpers.testF.cpp)
    target_link_libraries(testF helpers)
    add_test(NAME testF COMMAND testF)

    add_executable(testG helpers.testG.cpp)
    target_link_libraries(testG helpers)
    add_test(NAME testG COMMAND testG)
