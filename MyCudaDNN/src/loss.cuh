#ifndef LOSS_H_
#define LOSS_H_

#include "tensor.cuh"

namespace cudl {

    class CrossEntropyLoss {
    public:
        CrossEntropyLoss();

        ~CrossEntropyLoss();

        float loss(Tensor<float> *predict, Tensor<float> *target);

    private:
        // Reduced loss
        float h_loss_ = 0.f;
        float *d_loss_ = nullptr;
        float *d_workspace_ = nullptr;

        void init_workspace(int batch_size);
    };

} // namespace cudl

#endif // LOSS_H_