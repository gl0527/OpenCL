# setting the minimum version of CMake
cmake_minimum_required(VERSION 2.8)

# setting the name of the project
project(gol)

# setting build type as release
set(CMAKE_BUILD_TYPE Debug)

# setting the version of C++ standard
set (CMAKE_CXX_STANDARD 11)

# building the executable called game_of_life using the file called gol.cpp
add_executable(${CMAKE_PROJECT_NAME} gol.cpp)

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCL_LIBRARIES})

#set(CMAKE_CXX_FLAGS "-O0 -W -g -Wall -Wextra -pedantic")
#set(CMAKE_EXE_LINKER_FLAGS  "-lglut -lGL -lOpenCL")

