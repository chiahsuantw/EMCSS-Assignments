#ifndef MNIST_H_
#define MNIST_H_

#include <string>
#include <fstream>
#include <array>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>

#include "tensor.cuh"

namespace cudl {

#define MNIST_CLASS 10

    class MNIST {
    public:
        MNIST() : dataset_dir_("./") {}

        explicit MNIST(std::string dataset_dir) : dataset_dir_(std::move(dataset_dir)) {}

        ~MNIST();

        // load train dataset
        void train(int batch_size = 1, bool shuffle = false);

        // load test dataset
        void test(int batch_size = 1);

        void loadImage(const std::string& imagePath);

        // update shared batch data buffer at current step index
        void getBatch();

        // increase current step index
        // optionally it updates shared buffer if input parameter is true.
        int next();

        // returns a pointer which has input batch data
        Tensor<float> *getData() { return data_; }

        // returns a pointer which has target batch data
        Tensor<float> *getTarget() { return target_; }

    private:
        // predefined file names
        std::string dataset_dir_;

#ifdef __linux__
        std::string train_dataset_file_ = "train-images-idx3-ubyte";
        std::string train_label_file_   = "train-labels-idx1-ubyte";
        std::string test_dataset_file_  = "t10k-images-idx3-ubyte";
        std::string test_label_file_    = "t10k-labels-idx1-ubyte";
#elif _WIN32
        std::string train_dataset_file_ = "train-images.idx3-ubyte";
        std::string train_label_file_ = "train-labels.idx1-ubyte";
        std::string test_dataset_file_ = "t10k-images.idx3-ubyte";
        std::string test_label_file_ = "t10k-labels.idx1-ubyte";
#endif

        // container
        std::vector<std::vector<float>> data_pool_;
        std::vector<std::array<float, MNIST_CLASS>> target_pool_;
        Tensor<float> *data_ = nullptr;
        Tensor<float> *target_ = nullptr;

        // data loader initialization
        void loadData(std::string &image_file_path);

        void loadTarget(std::string &label_file_path);

        static int toInt(const uint8_t *ptr);

        // data loader control
        int step_ = -1;
        bool shuffle_ = true;
        int batch_size_ = 1;
        int channels_ = 1;
        int height_ = 1;
        int width_ = 1;
        int num_classes_ = 10;
        int num_steps_ = 0;

        void createSharedSpace();

        void shuffleDataset();
    };

} // namespace cudl

#endif // MNIST_H_
