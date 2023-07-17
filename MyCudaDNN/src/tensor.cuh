#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace cudl {
    typedef enum {
        host,
        cuda
    } DeviceType;

    template<typename type>
    class Tensor {
    public:
        explicit Tensor(int n = 1, int c = 1, int h = 1, int w = 1) : n_(n), c_(c), h_(h), w_(w) {
            h_ptr_ = new float[n_ * c_ * h_ * w_];
        }

        explicit Tensor(std::array<int, 4> size) : n_(size[0]), c_(size[1]), h_(size[2]), w_(size[3]) {
            h_ptr_ = new float[n_ * c_ * h_ * w_];
        }

        ~Tensor() {
            if (h_ptr_ != nullptr)
                delete[] h_ptr_;
            if (d_ptr_ != nullptr)
                cudaFree(d_ptr_);
            if (is_tensor_)
                cudnnDestroyTensorDescriptor(tensor_desc_);
        }

        // Reset the current blob with new size
        void reset(int n = 1, int c = 1, int h = 1, int w = 1) {
            // Update size
            n_ = n;
            c_ = c;
            h_ = h;
            w_ = w;

            // Terminate current buffers
            if (h_ptr_ != nullptr) {
                delete[] h_ptr_;
                h_ptr_ = nullptr;
            }
            if (d_ptr_ != nullptr) {
                cudaFree(d_ptr_);
                d_ptr_ = nullptr;
            }

            // Create new buffer
            h_ptr_ = new float[n_ * c_ * h_ * w_];
            cuda();

            // Reset tensor descriptor if it was a tensor
            if (is_tensor_) {
                cudnnDestroyTensorDescriptor(tensor_desc_);
                is_tensor_ = false;
            }
        }

        void reset(std::array<int, 4> size) {
            reset(size[0], size[1], size[2], size[3]);
        }

        // Return the array of tensor shape
        std::array<int, 4> shape() { return std::array<int, 4>({n_, c_, h_, w_}); }

        // Return the number of elements for 1 batch
        int size() { return c_ * h_ * w_; }

        // Return the number of total elements in blob including batch
        int len() { return n_ * c_ * h_ * w_; }

        // Return the size of allocated memory
        int buf_size() { return sizeof(type) * len(); }

        [[nodiscard]] int n() const { return n_; }

        [[nodiscard]] int c() const { return c_; }

        [[nodiscard]] int h() const { return h_; }

        [[nodiscard]] int w() const { return w_; }

        /* Tensor Control */
        bool is_tensor_ = false;
        cudnnTensorDescriptor_t tensor_desc_{};

        cudnnTensorDescriptor_t tensor() {
            if (is_tensor_)
                return tensor_desc_;

            cudnnCreateTensorDescriptor(&tensor_desc_);
            cudnnSetTensor4dDescriptor(tensor_desc_,
                                       CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       n_, c_, h_, w_);
            is_tensor_ = true;

            return tensor_desc_;
        }

        /* Memory Control */
        // Get specified memory pointer
        type *ptr() { return h_ptr_; }

        // Get cuda memory
        type *cuda() {
            if (d_ptr_ == nullptr)
                cudaMalloc((void **) &d_ptr_, sizeof(type) * len());
            return d_ptr_;
        }

        // Transfer data between memory
        type *to(DeviceType target) {
            type *ptr;
            if (target == host) {
                cudaMemcpy(h_ptr_, cuda(), sizeof(type) * len(), cudaMemcpyDeviceToHost);
                ptr = h_ptr_;
            } else {
                cudaMemcpy(cuda(), h_ptr_, sizeof(type) * len(), cudaMemcpyHostToDevice);
                ptr = d_ptr_;
            }
            return ptr;
        }

        /* Pretrained parameter load and save */
        int file_read(const std::string &filename) {
            std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
            if (!file.is_open()) {
                std::cout << "failed to access " << filename << std::endl;
                return -1;
            }

            file.read((char *) h_ptr_, sizeof(float) * this->len());
            this->to(DeviceType::cuda);
            file.close();

            return 0;
        }

        int file_write(const std::string &filename) {
            std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                std::cout << "failed to write " << filename << std::endl;
                return -1;
            }
            file.write((char *) this->to(host), sizeof(float) * this->len());
            file.close();

            return 0;
        }

    private:
        type *h_ptr_ = nullptr;
        type *d_ptr_ = nullptr;

        int n_ = 1;
        int c_ = 1;
        int h_ = 1;
        int w_ = 1;
    };

} // namespace cudl

#endif // TENSOR_H_