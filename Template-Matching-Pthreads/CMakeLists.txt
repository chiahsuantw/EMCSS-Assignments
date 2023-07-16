cmake_minimum_required(VERSION 3.25)
project(pthread)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS -pthread)

add_executable(pthread main.cpp)
