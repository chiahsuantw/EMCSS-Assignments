#ifndef NETWORK_H_
#define NETWORK_H_

#include <string>
#include <vector>
#include <cudnn.h>

#include "utils.cuh"
#include "loss.cuh"
#include "layer.cuh"

namespace cudl {
    typedef enum {
        training,
        inference
    } WorkloadType;

    class Network {
    public:
        Network();

        ~Network();

        void addLayer(Layer *layer);

        Tensor<float> *forward(Tensor<float> *input);

        void backward(Tensor<float> *input = nullptr);

        void update(float learning_rate = 0.02f);

        int loadPretrain();

        int writeFile();

        float loss(Tensor<float> *target);

        int getAccuracy(Tensor<float> *target);

        void cuda();

        void train();

        void test();

        int predict(Tensor<float> *input);

        Tensor<float> *output_{};

    private:
        std::vector<Layer *> layers_;
        CudaContext *cuda_ = nullptr;
        WorkloadType phase_ = inference;
    };

} // namespace cudl

#endif // NETWORK_H_