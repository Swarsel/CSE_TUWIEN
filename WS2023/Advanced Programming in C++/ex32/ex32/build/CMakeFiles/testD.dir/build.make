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
include CMakeFiles/testD.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/testD.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/testD.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testD.dir/flags.make

CMakeFiles/testD.dir/unique_ptr.testD.cpp.o: CMakeFiles/testD.dir/flags.make
CMakeFiles/testD.dir/unique_ptr.testD.cpp.o: /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testD.cpp
CMakeFiles/testD.dir/unique_ptr.testD.cpp.o: CMakeFiles/testD.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/swarsel/Documents/GitHub/ex32/ex32/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testD.dir/unique_ptr.testD.cpp.o"
	/nix/store/sfgnb6rr428bssyrs54d6d0vv2avi95c-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testD.dir/unique_ptr.testD.cpp.o -MF CMakeFiles/testD.dir/unique_ptr.testD.cpp.o.d -o CMakeFiles/testD.dir/unique_ptr.testD.cpp.o -c /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testD.cpp

CMakeFiles/testD.dir/unique_ptr.testD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testD.dir/unique_ptr.testD.cpp.i"
	/nix/store/sfgnb6rr428bssyrs54d6d0vv2avi95c-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testD.cpp > CMakeFiles/testD.dir/unique_ptr.testD.cpp.i

CMakeFiles/testD.dir/unique_ptr.testD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testD.dir/unique_ptr.testD.cpp.s"
	/nix/store/sfgnb6rr428bssyrs54d6d0vv2avi95c-gcc-wrapper-12.3.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swarsel/Documents/GitHub/ex32/ex32/unique_ptr.testD.cpp -o CMakeFiles/testD.dir/unique_ptr.testD.cpp.s

# Object files for target testD
testD_OBJECTS = \
"CMakeFiles/testD.dir/unique_ptr.testD.cpp.o"

# External object files for target testD
testD_EXTERNAL_OBJECTS =

testD: CMakeFiles/testD.dir/unique_ptr.testD.cpp.o
testD: CMakeFiles/testD.dir/build.make
testD: CMakeFiles/testD.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/swarsel/Documents/GitHub/ex32/ex32/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testD"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testD.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testD.dir/build: testD
.PHONY : CMakeFiles/testD.dir/build

CMakeFiles/testD.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testD.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testD.dir/clean

CMakeFiles/testD.dir/depend:
	cd /home/swarsel/Documents/GitHub/ex32/ex32/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/swarsel/Documents/GitHub/ex32/ex32 /home/swarsel/Documents/GitHub/ex32/ex32 /home/swarsel/Documents/GitHub/ex32/ex32/build /home/swarsel/Documents/GitHub/ex32/ex32/build /home/swarsel/Documents/GitHub/ex32/ex32/build/CMakeFiles/testD.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/testD.dir/depend

