cmake_minimum_required(VERSION 3.12)

# header only library

    add_library(List INTERFACE)
    target_sources(List INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/List.hpp)       

# tests

    add_executable(testA List.testA.cpp)
    add_test(NAME testA COMMAND testA)

    add_executable(testB List.testB.cpp)
    add_test(NAME testB COMMAND testB)

    add_executable(testC List.testC.cpp)
    add_test(NAME testC COMMAND testC)

    add_executable(testD List.testD.cpp)
    add_test(NAME testD COMMAND testD)

    add_executable(testE List.testE.cpp)
    add_test(NAME testE COMMAND testE)

    add_executable(testF List.testF.cpp)
    add_test(NAME testF COMMAND testF)

    add_executable(testG List.testG.cpp)
    add_test(NAME testG COMMAND testG)

    add_executable(testH List.testH.cpp)
    add_test(NAME testH COMMAND testH)

