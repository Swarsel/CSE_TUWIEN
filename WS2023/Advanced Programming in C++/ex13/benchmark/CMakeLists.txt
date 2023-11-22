cmake_minimum_required(VERSION 3.12)

# excutables

    add_executable(benchmark)
    target_sources(benchmark PRIVATE main.cpp)    
    target_link_libraries(benchmark PRIVATE List)
    target_include_directories(benchmark SYSTEM PRIVATE ${CMAKE_SOURCE_DIR})
