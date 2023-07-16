cmake_minimum_required(VERSION 3.25)
project(EMCSS_Project_1 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(EMCSS_Project_1 main.cu)

set_target_properties(EMCSS_Project_1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
