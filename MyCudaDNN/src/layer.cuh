#ifndef LAYER_H_
#define LAYER_H_

#include <string>
#include <cublas_v2.h>
#include <cudnn.h>

#include "tensor.cuh"
#include "loss.cuh"
#include "utils.cuh"

namespace cudl {
    class Layer {
    public:
        Layer();

        virtual ~Layer();

        virtual Tensor<float> *forward(Tensor<float> *input) = 0;

        virtual Tensor<float> *backward(Tensor<float> *grad_input) = 0;

        std::string get_name() { return name_; }

        virtual float get_loss(Tensor<float> *target);

        virtual int get_accuracy(Tensor<float> *target);

        void set_cuda_context(CudaContext *context) { cuda_ = context; }

        void set_load_pretrain() { load_pretrain_ = true; };

        void set_gradient_stop() { gradient_stop_ = true; }

        /* Weight freeze or unfreeze */
        void freeze() { freeze_ = true; }

        void unfreeze() { freeze_ = false; }

    protected:
        virtual void fwd_initialize(Tensor<float> *input) = 0;

        virtual void bwd_initialize(Tensor<float> *grad_output) = 0;

        // Name of the layer
        std::string name_;

        // Tensor descriptor for the input/output tensors
        cudnnTensorDescriptor_t input_desc_{};
        cudnnTensorDescriptor_t output_desc_{};

        // Output memory
        Tensor<float> *input_ = nullptr;         /* x  */
        Tensor<float> *output_ = nullptr;        /* y  */
        Tensor<float> *grad_input_ = nullptr;    /* dx */
        Tensor<float> *grad_output_ = nullptr;   /* dy */

        // Master weights & bias
        bool freeze_ = false; // control parameter updates
        Tensor<float> *weights_ = nullptr;       /* w */
        Tensor<float> *biases_ = nullptr;        /* b */
        Tensor<float> *grad_weights_ = nullptr;  /* dw */
        Tensor<float> *grad_biases_ = nullptr;   /* db */

        int batch_size_ = 0;  // mini-batch size

        // Initialize weights along with the input size
        void init_weight_bias(unsigned int seed = 0);

        void update_weights_biases(float learning_rate);

        // Cuda handle container
        CudaContext *cuda_ = nullptr;

        // Pretrain parameters
        bool load_pretrain_ = false;

        int load_parameter();

        int save_parameter();

        // Gradient stop tagging
        bool gradient_stop_ = false;

        friend class Network;
    };

    class Dense : public Layer {
    public:
        Dense(std::string name, int out_size);

        ~Dense() override;

        Tensor<float> *forward(Tensor<float> *input) override;

        Tensor<float> *backward(Tensor<float> *grad_input) override;

    private:
        void fwd_initialize(Tensor<float> *input) override;

        void bwd_initialize(Tensor<float> *grad_output) override;

        int input_size_ = 0;
        int output_size_ = 0;

        float *d_one_vec = nullptr;
    };

    class Activation : public Layer {
    public:
        Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);

        ~Activation() override;

        Tensor<float> *forward(Tensor<float> *input) override;

        Tensor<float> *backward(Tensor<float> *grad_input) override;

    private:
        void fwd_initialize(Tensor<float> *input) override;

        void bwd_initialize(Tensor<float> *grad_output) override;

        cudnnActivationDescriptor_t act_desc_{};
        cudnnActivationMode_t act_mode_;
        float act_coef_;
    };

    class Softmax : public Layer {
    public:
        explicit Softmax(std::string name);

        ~Softmax() override;

        Tensor<float> *forward(Tensor<float> *input) override;

        Tensor<float> *backward(Tensor<float> *grad_input) override;

        float get_loss(Tensor<float> *target) override;

        int get_accuracy(Tensor<float> *target) override;

    protected:
        void fwd_initialize(Tensor<float> *input) override;

        void bwd_initialize(Tensor<float> *grad_output) override;

        CrossEntropyLoss loss_;
    };

} // namespace cudl

#endif // LAYER_H_