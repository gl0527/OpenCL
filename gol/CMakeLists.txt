# setting the minimum version of CMake
cmake_minimum_required (VERSION 2.8)

# setting the name of the project
project (game_of_life)

# setting build type as debug
set (CMAKE_BUILD_TYPE Debug)

# setting the compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -std=c++11 -W -Wall -Wextra -pedantic -Wno-long-long")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
    endif ()
endif (CMAKE_COMPILER_IS_GNUCC)

# search for needed packages
find_package (OpenCL REQUIRED)
find_package (OpenGL REQUIRED)
find_package (GLUT REQUIRED)

# setting the include directories
include_directories (${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

# setting up executable
set (EXECUTABLE_NAME runnable)
add_executable (${EXECUTABLE_NAME} gol.cpp)
target_link_libraries (${EXECUTABLE_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCL_LIBRARIES})

