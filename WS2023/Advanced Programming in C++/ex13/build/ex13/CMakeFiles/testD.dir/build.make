# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /nix/store/vnhl4zdy7igx9gd3q1d548vwzz15a9ma-cmake-3.27.7/bin/cmake

# The command to remove a file.
RM = /nix/store/vnhl4zdy7igx9gd3q1d548vwzz15a9ma-cmake-3.27.7/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/swarsel/Documents/GitHub/ex13

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/swarsel/Documents/GitHub/ex13/build

# Include any dependencies generated for this target.
include ex13/CMakeFiles/testD.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ex13/CMakeFiles/testD.dir/compiler_depend.make

# Include the progress variables for this target.
include ex13/CMakeFiles/testD.dir/progress.make

# Include the compile flags for this target's objects.
include ex13/CMakeFiles/testD.dir/flags.make

ex13/CMakeFiles/testD.dir/List.testD.cpp.o: ex13/CMakeFiles/testD.dir/flags.make
ex13/CMakeFiles/testD.dir/List.testD.cpp.o: /home/swarsel/Documents/GitHub/ex13/ex13/List.testD.cpp
ex13/CMakeFiles/testD.dir/List.testD.cpp.o: ex13/CMakeFiles/testD.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/swarsel/Documents/GitHub/ex13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ex13/CMakeFiles/testD.dir/List.testD.cpp.o"
	cd /home/swarsel/Documents/GitHub/ex13/build/ex13 && /nix/store/90h6k8ylkgn81k10190v5c9ldyjpzgl9-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ex13/CMakeFiles/testD.dir/List.testD.cpp.o -MF CMakeFiles/testD.dir/List.testD.cpp.o.d -o CMakeFiles/testD.dir/List.testD.cpp.o -c /home/swarsel/Documents/GitHub/ex13/ex13/List.testD.cpp

ex13/CMakeFiles/testD.dir/List.testD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testD.dir/List.testD.cpp.i"
	cd /home/swarsel/Documents/GitHub/ex13/build/ex13 && /nix/store/90h6k8ylkgn81k10190v5c9ldyjpzgl9-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swarsel/Documents/GitHub/ex13/ex13/List.testD.cpp > CMakeFiles/testD.dir/List.testD.cpp.i

ex13/CMakeFiles/testD.dir/List.testD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testD.dir/List.testD.cpp.s"
	cd /home/swarsel/Documents/GitHub/ex13/build/ex13 && /nix/store/90h6k8ylkgn81k10190v5c9ldyjpzgl9-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swarsel/Documents/GitHub/ex13/ex13/List.testD.cpp -o CMakeFiles/testD.dir/List.testD.cpp.s

# Object files for target testD
testD_OBJECTS = \
"CMakeFiles/testD.dir/List.testD.cpp.o"

# External object files for target testD
testD_EXTERNAL_OBJECTS =

ex13/testD: ex13/CMakeFiles/testD.dir/List.testD.cpp.o
ex13/testD: ex13/CMakeFiles/testD.dir/build.make
ex13/testD: /nix/store/qn3ggz5sf3hkjs2c797xf7nan3amdxmp-glibc-2.38-27/lib/libm.so
ex13/testD: ex13/libList.so
ex13/testD: /nix/store/qn3ggz5sf3hkjs2c797xf7nan3amdxmp-glibc-2.38-27/lib/libm.so
ex13/testD: ex13/CMakeFiles/testD.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/swarsel/Documents/GitHub/ex13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testD"
	cd /home/swarsel/Documents/GitHub/ex13/build/ex13 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testD.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ex13/CMakeFiles/testD.dir/build: ex13/testD
.PHONY : ex13/CMakeFiles/testD.dir/build

ex13/CMakeFiles/testD.dir/clean:
	cd /home/swarsel/Documents/GitHub/ex13/build/ex13 && $(CMAKE_COMMAND) -P CMakeFiles/testD.dir/cmake_clean.cmake
.PHONY : ex13/CMakeFiles/testD.dir/clean

ex13/CMakeFiles/testD.dir/depend:
	cd /home/swarsel/Documents/GitHub/ex13/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/swarsel/Documents/GitHub/ex13 /home/swarsel/Documents/GitHub/ex13/ex13 /home/swarsel/Documents/GitHub/ex13/build /home/swarsel/Documents/GitHub/ex13/build/ex13 /home/swarsel/Documents/GitHub/ex13/build/ex13/CMakeFiles/testD.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : ex13/CMakeFiles/testD.dir/depend

