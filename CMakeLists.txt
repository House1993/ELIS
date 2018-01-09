cmake_minimum_required(VERSION 3.6)

project(ELIS)

set(CMAKE_CXX_STANDARD 11)

## Install the executable to bin fodler
set(EXECUTABLE_OUTPUT_PATH run)

add_executable(discover tfidfinit.cpp)
add_executable(adjust learn.cpp)
