cmake_minimum_required(VERSION 3.12)

# define project metadata

    project(E360251 LANGUAGES CXX
                    DESCRIPTION "ex22"
                    HOMEPAGE_URL "https://sgit.iue.tuwien.ac.at/360251/ex22")

# setting required language standard

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

# enable export of compile commands to compile_commands.json

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# enable system includes for each source compiled with the CXX language

    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# platform specific options compile options

    if(MSVC)
        add_compile_options(-W4 -WX) # warnings as errors
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # make symbols in shared libs extern by default
    elseif(UNIX)
        add_compile_options(-Wall -Werror -pedantic) # warnings as errors; strict mode
    endif()

# configure CMAKE_BUILD_TYPE=DEBUG to use AddressSanitizer (ASAN) for supported compilers

    set(FLAG "-fsanitize=address")
    include(CheckCXXCompilerFlag)
    set(CMAKE_REQUIRED_FLAGS ${FLAG})
    unset(COMPILER_SUPPORTS_ASAN)
    check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORTS_ASAN)

    if (COMPILER_SUPPORTS_ASAN)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} ${FLAG}" CACHE STRING "" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG_INIT} ${FLAG}" CACHE STRING "" FORCE)
        set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG_INIT} ${FLAG}" CACHE STRING "" FORCE)
    endif()

# find math library and link to all targets as default

    find_library(MATH_LIBRARY m)
    if(MATH_LIBRARY)
        link_libraries(${MATH_LIBRARY})
    endif()

# enable CTest

    enable_testing()

# including targets

    add_subdirectory(ex22)
    add_subdirectory(benchmark)    

# build documentation (requires doxygen)

    option(BUILD_DOC "Build documentation" OFF)
    if(BUILD_DOC)
        add_subdirectory(doxygen)
    endif()
