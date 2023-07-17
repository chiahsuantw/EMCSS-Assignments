#ifndef HELPER_H_
#define HELPER_H_

#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>

namespace cudl {

#define BLOCK_DIM_1D 512

    class CudaContext {
    public:
        CudaContext() {
            cublasCreate(&_cublas_handle);
            cudnnCreate(&_cudnn_handle);
        }

        ~CudaContext() {
            cublasDestroy(_cublas_handle);
            cudnnDestroy(_cudnn_handle);
        }

        cublasHandle_t cublas() {
            return _cublas_handle;
        };

        cudnnHandle_t cudnn() { return _cudnn_handle; };

        const float one = 1.f;
        const float zero = 0.f;
        const float minus_one = -1.f;

    private:
        cublasHandle_t _cublas_handle{};
        cudnnHandle_t _cudnn_handle{};
    };

} // namespace cudl

#endif // HELPER_H_