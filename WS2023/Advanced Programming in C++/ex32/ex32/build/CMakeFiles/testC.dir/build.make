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
CMAKE_COMMAND = /nix/store/4vq5ggsg1vmfs09r4sqbidmgvqlxrv14-cmake-3.27.8/bin/cmake

# The command to remove a file.
RM = /nix/store/4vq5ggsg1vmfs09r4sqbidmgvqlxrv14-cmake-3.27.8/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/swarsel/Documents/GitHub/ex32/ex32

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/swarsel/Documents/GitHub/ex32/ex32/build

# Include any dependencies generated for this target.
include CMakeFiles/testC.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/testC.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/testC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testC.dir/flags.make

CMakeFiles/testC.dir/unique_ptr.testC.cpp.o: CMakeFiles/testC.dir/flags.make
CMakeFiles/testC.dir/unique_ptr.testC.cpp.o: /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testC.cpp
CMakeFiles/testC.dir/unique_ptr.testC.cpp.o: CMakeFiles/testC.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/swarsel/Documents/GitHub/ex32/ex32/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testC.dir/unique_ptr.testC.cpp.o"
	/nix/store/sfgnb6rr428bssyrs54d6d0vv2avi95c-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testC.dir/unique_ptr.testC.cpp.o -MF CMakeFiles/testC.dir/unique_ptr.testC.cpp.o.d -o CMakeFiles/testC.dir/unique_ptr.testC.cpp.o -c /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testC.cpp

CMakeFiles/testC.dir/unique_ptr.testC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testC.dir/unique_ptr.testC.cpp.i"
	/nix/store/sfgnb6rr428bssyrs54d6d0vv2avi95c-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testC.cpp > CMakeFiles/testC.dir/unique_ptr.testC.cpp.i

CMakeFiles/testC.dir/unique_ptr.testC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testC.dir/unique_ptr.testC.cpp.s"
	/nix/store/sfgnb6rr428bssyrs54d6d0vv2avi95c-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testC.cpp -o CMakeFiles/testC.dir/unique_ptr.testC.cpp.s

# Object files for target testC
testC_OBJECTS = \
"CMakeFiles/testC.dir/unique_ptr.testC.cpp.o"

# External object files for target testC
testC_EXTERNAL_OBJECTS =

testC: CMakeFiles/testC.dir/unique_ptr.testC.cpp.o
testC: CMakeFiles/testC.dir/build.make
testC: CMakeFiles/testC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/swarsel/Documents/GitHub/ex32/ex32/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testC.dir/build: testC
.PHONY : CMakeFiles/testC.dir/build

CMakeFiles/testC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testC.dir/clean

CMakeFiles/testC.dir/depend:
	cd /home/swarsel/Documents/GitHub/ex32/ex32/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/swarsel/Documents/GitHub/ex32/ex32 /home/swarsel/Documents/GitHub/ex32/ex32 /home/swarsel/Documents/GitHub/ex32/ex32/build /home/swarsel/Documents/GitHub/ex32/ex32/build /home/swarsel/Documents/GitHub/ex32/ex32/build/CMakeFiles/testC.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/testC.dir/depend

