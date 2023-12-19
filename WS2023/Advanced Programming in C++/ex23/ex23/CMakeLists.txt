cmake_minimum_required(VERSION 3.12)

# header only library

    add_library(Algorithms INTERFACE)
    target_sources(Algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Algorithms.hpp)       

# tests

    add_executable(testA Algorithms.testA.cpp)
    add_test(NAME testA COMMAND testA)

    add_executable(testB Algorithms.testB.cpp)
    add_test(NAME testB COMMAND testB)

    add_executable(testC Algorithms.testC.cpp)
    add_test(NAME testC COMMAND testC)

    add_executable(testD Algorithms.testD.cpp)
    add_test(NAME testD COMMAND testD)

    add_executable(testE Algorithms.testE.cpp)
    add_test(NAME testE COMMAND testE)

    add_executable(testF Algorithms.testF.cpp)
    add_test(NAME testF COMMAND testF)

    add_executable(testG Algorithms.testG.cpp)
    add_test(NAME testG COMMAND testG)

    add_executable(testH Algorithms.testH.cpp)
    add_test(NAME testH COMMAND testH)

    add_executable(testI Algorithms.testI.cpp)
    add_test(NAME testI COMMAND testI)



