# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\dj5jgf\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7846.88\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\dj5jgf\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\201.7846.88\bin\cmake\win\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\dj5jgf\Desktop\LeetCode

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/LeetCode.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LeetCode.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LeetCode.dir/flags.make

CMakeFiles/LeetCode.dir/main.cpp.obj: CMakeFiles/LeetCode.dir/flags.make
CMakeFiles/LeetCode.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LeetCode.dir/main.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\LeetCode.dir\main.cpp.obj -c C:\Users\dj5jgf\Desktop\LeetCode\main.cpp

CMakeFiles/LeetCode.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LeetCode.dir/main.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\dj5jgf\Desktop\LeetCode\main.cpp > CMakeFiles\LeetCode.dir\main.cpp.i

CMakeFiles/LeetCode.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LeetCode.dir/main.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\dj5jgf\Desktop\LeetCode\main.cpp -o CMakeFiles\LeetCode.dir\main.cpp.s

CMakeFiles/LeetCode.dir/1_TwoSum.cpp.obj: CMakeFiles/LeetCode.dir/flags.make
CMakeFiles/LeetCode.dir/1_TwoSum.cpp.obj: ../1_TwoSum.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/LeetCode.dir/1_TwoSum.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\LeetCode.dir\1_TwoSum.cpp.obj -c C:\Users\dj5jgf\Desktop\LeetCode\1_TwoSum.cpp

CMakeFiles/LeetCode.dir/1_TwoSum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LeetCode.dir/1_TwoSum.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\dj5jgf\Desktop\LeetCode\1_TwoSum.cpp > CMakeFiles\LeetCode.dir\1_TwoSum.cpp.i

CMakeFiles/LeetCode.dir/1_TwoSum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LeetCode.dir/1_TwoSum.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\dj5jgf\Desktop\LeetCode\1_TwoSum.cpp -o CMakeFiles\LeetCode.dir\1_TwoSum.cpp.s

# Object files for target LeetCode
LeetCode_OBJECTS = \
"CMakeFiles/LeetCode.dir/main.cpp.obj" \
"CMakeFiles/LeetCode.dir/1_TwoSum.cpp.obj"

# External object files for target LeetCode
LeetCode_EXTERNAL_OBJECTS =

LeetCode.exe: CMakeFiles/LeetCode.dir/main.cpp.obj
LeetCode.exe: CMakeFiles/LeetCode.dir/1_TwoSum.cpp.obj
LeetCode.exe: CMakeFiles/LeetCode.dir/build.make
LeetCode.exe: CMakeFiles/LeetCode.dir/linklibs.rsp
LeetCode.exe: CMakeFiles/LeetCode.dir/objects1.rsp
LeetCode.exe: CMakeFiles/LeetCode.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable LeetCode.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\LeetCode.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LeetCode.dir/build: LeetCode.exe

.PHONY : CMakeFiles/LeetCode.dir/build

CMakeFiles/LeetCode.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\LeetCode.dir\cmake_clean.cmake
.PHONY : CMakeFiles/LeetCode.dir/clean

CMakeFiles/LeetCode.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\dj5jgf\Desktop\LeetCode C:\Users\dj5jgf\Desktop\LeetCode C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug C:\Users\dj5jgf\Desktop\LeetCode\cmake-build-debug\CMakeFiles\LeetCode.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LeetCode.dir/depend
