cmake_minimum_required(VERSION 3.25)
project(MyCudaDNN CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(MyCudaDNN
        train.cu
        src/tensor.cuh
        src/utils.cuh
        src/layer.cu src/layer.cuh
        src/loss.cu src/loss.cuh
        src/mnist.cu src/mnist.cuh
        src/network.cu src/network.cuh)

# cuBLAS and cuDNN
target_link_libraries(MyCudaDNN cublas.lib cudnn.lib)

# OpenCV
set(OpenCV_DIR "D:\\Program Files\\opencv\\build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(MyCudaDNN ${OpenCV_LIBS})

add_definitions(-DUNICODE -D_UNICODE)

set_target_properties(MyCudaDNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
