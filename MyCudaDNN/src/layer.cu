#include <random>
#include <cassert>
#include <cmath>
#include <sstream>
#include <iostream>
#include <utility>

#include "layer.cuh"

using namespace cudl;

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/
Layer::Layer() = default;

Layer::~Layer() {
    if (output_ != nullptr) {
        delete output_;
        output_ = nullptr;
    }
    if (grad_input_ != nullptr) {
        delete grad_input_;
        grad_input_ = nullptr;
    }

    if (weights_ != nullptr) {
        delete weights_;
        weights_ = nullptr;
    }
    if (biases_ != nullptr) {
        delete biases_;
        biases_ = nullptr;
    }
    if (grad_weights_ != nullptr) {
        delete grad_weights_;
        grad_weights_ = nullptr;
    }
    if (grad_biases_ != nullptr) {
        delete grad_biases_;
        grad_biases_ = nullptr;
    }
}

void Layer::init_weight_bias(unsigned int seed) {
    cudaDeviceSynchronize();

    if (weights_ == nullptr || biases_ == nullptr)
        return;

    // Create random network
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    // He uniform distribution
    float range = sqrt(6.f / (float) input_->size()); // He's initialization
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
        weights_->ptr()[i] = static_cast<float>(dis(gen));
    for (int i = 0; i < biases_->len(); i++)
        biases_->ptr()[i] = 0.f;

    // Copy initialized value to the device
    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);

    // std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}

void Layer::update_weights_biases(float learning_rate) {
    float eps = -1.f * learning_rate;
    if (weights_ != nullptr && grad_weights_ != nullptr) {
        // w = w + eps * dw
        cublasSaxpy(cuda_->cublas(),
                    weights_->len(),
                    &eps,
                    grad_weights_->cuda(), 1,
                    weights_->cuda(), 1);
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
        // b = b + eps * db
        cublasSaxpy(cuda_->cublas(),
                    biases_->len(),
                    &eps,
                    grad_biases_->cuda(), 1,
                    biases_->cuda(), 1);
    }
}

float Layer::get_loss(Tensor<float> *target) {
    assert("No Loss layer has no loss." && false);
    return EXIT_FAILURE;
}

int Layer::get_accuracy(Tensor<float> *target) {
    assert("No Loss layer cannot estimate accuracy." && false);
    return EXIT_FAILURE;
}

int Layer::load_parameter() {
    std::stringstream filename_weights, filename_biases;

    // load weights and biases pretrained parameters
    filename_weights << name_ << ".bin";
    if (weights_->file_read(filename_weights.str()))
        return -1;

    filename_biases << name_ << ".bias.bin";
    if (biases_->file_read(filename_biases.str()))
        return -2;

    std::cout << ".. loaded " << name_ << " pretrain parameter.." << std::endl;

    return 0;
}

int Layer::save_parameter() {
    std::stringstream filename_weights, filename_biases;

    std::cout << "Saving " << name_ << " parameter: ";

    // Write weights file
    if (weights_) {
        filename_weights << name_ << ".bin";
        if (weights_->file_write(filename_weights.str()))
            return -1;
    }

    // Write bias file
    if (biases_) {
        filename_biases << name_ << ".bias.bin";
        if (biases_->file_write(filename_biases.str()))
            return -2;
    }

    std::cout << "done" << std::endl;

    return 0;
}

/****************************************************************
 * Dense Layer                                                  *
 ****************************************************************/

Dense::Dense(std::string name, int output_size) {
    name_ = std::move(name);
    output_size_ = output_size;
}

Dense::~Dense() {
    if (d_one_vec != nullptr) {
        cudaFree(d_one_vec);
        d_one_vec = nullptr;
    }
}

__global__ void init_one_vec(float *d_one_vec, size_t length) {
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= length) return;

    d_one_vec[i] = 1.f;
}

void Dense::fwd_initialize(Tensor<float> *input) {
    // initialize weights and biases
    if (weights_ == nullptr) {
        // setup parameter size information
        input_size_ = input->c() * input->h() * input->w();

        // initialize weight, bias, and output
        weights_ = new Tensor<float>(1, 1, input_size_, output_size_);
        biases_ = new Tensor<float>(1, 1, output_size_);
    }

    // Initialize input and output
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Tensor<float>(batch_size_, output_size_);
        else
            output_->reset(batch_size_, output_size_);

        output_->tensor();

        if (d_one_vec != nullptr)
            cudaFree(d_one_vec);
        cudaMalloc((void **) &d_one_vec, sizeof(float) * batch_size_);
        init_one_vec<<< (batch_size_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec, batch_size_);

        // Initialize weights and biases
        if (load_pretrain_ && !freeze_) {
            if (load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!freeze_) {
            init_weight_bias();
        }
    }
}

Tensor<float> *Dense::forward(Tensor<float> *input) {
    // output = weights^T * input (without biases)
    cublasSgemm(cuda_->cublas(),
                CUBLAS_OP_T, CUBLAS_OP_N,
                output_size_, batch_size_, input_size_,
                &cuda_->one,
                weights_->cuda(), input_size_,
                input_->cuda(), input_size_,
                &cuda_->zero,
                output_->cuda(), output_size_);

    // output += biases * d_one_vec^T
    cublasSgemm(cuda_->cublas(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                output_size_, batch_size_, 1,
                &cuda_->one,
                biases_->cuda(), output_size_,
                d_one_vec, 1,
                &cuda_->one,
                output_->cuda(), output_size_);

    return output_;
}

void Dense::bwd_initialize(Tensor<float> *grad_output) {
    if (grad_weights_ == nullptr) {
        grad_weights_ = new Tensor<float>(weights_->shape());
        grad_biases_ = new Tensor<float>(biases_->shape());
    }

    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr)
            grad_input_ = new Tensor<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
}

Tensor<float> *Dense::backward(Tensor<float> *grad_output) {
    // db = dy * one_vec
    cublasSgemv(cuda_->cublas(),
                CUBLAS_OP_N,
                output_size_, batch_size_,
                &cuda_->one,
                grad_output_->cuda(), output_size_,
                d_one_vec, 1,
                &cuda_->zero,
                grad_biases_->cuda(), 1);

    // dw = x * dy^T
    cublasSgemm(cuda_->cublas(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                input_size_, output_size_, batch_size_,
                &cuda_->one,
                input_->cuda(), input_size_,
                grad_output_->cuda(), output_size_,
                &cuda_->zero,
                grad_weights_->cuda(), input_size_);

    // dx = w * dy
    if (!gradient_stop_)
        cublasSgemm(cuda_->cublas(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    input_size_, batch_size_, output_size_,
                    &cuda_->one,
                    weights_->cuda(), input_size_,
                    grad_output_->cuda(), output_size_,
                    &cuda_->zero,
                    grad_input_->cuda(), input_size_);

    return grad_input_;
}

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/

Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef) {
    name_ = std::move(name);
    act_mode_ = mode;
    act_coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

Activation::~Activation() {
    cudnnDestroyActivationDescriptor(act_desc_);
}

void Activation::fwd_initialize(Tensor<float> *input) {
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Tensor<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();
    }
}

Tensor<float> *Activation::forward(Tensor<float> *input) {
    cudnnActivationForward(cuda_->cudnn(),
                           act_desc_,
                           &cuda_->one,
                           input_desc_,
                           input->cuda(),
                           &cuda_->zero,
                           output_desc_,
                           output_->cuda());

    return output_;
}

void Activation::bwd_initialize(Tensor<float> *grad_output) {
    if (grad_input_ == nullptr || batch_size_ != grad_output->n()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr)
            grad_input_ = new Tensor<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
}

Tensor<float> *Activation::backward(Tensor<float> *grad_output) {
    cudnnActivationBackward(cuda_->cudnn(),
                            act_desc_,
                            &cuda_->one,
                            output_desc_, output_->cuda(),
                            output_desc_, grad_output->cuda(),
                            input_desc_, input_->cuda(),
                            &cuda_->zero,
                            input_desc_, grad_input_->cuda());

    return grad_input_;
}

/****************************************************************
 * Softmax definition                                           *
 ****************************************************************/

Softmax::Softmax(std::string name) {
    name_ = std::move(name);
}

Softmax::~Softmax() = default;

void Softmax::fwd_initialize(Tensor<float> *input) {
    if (input_ == nullptr || batch_size_ != input->n()) {
        input_ = input;
        input_desc_ = input->tensor();
        batch_size_ = input->n();

        if (output_ == nullptr)
            output_ = new Tensor<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor();
    }
}

Tensor<float> *Softmax::forward(Tensor<float> *input) {
    cudnnSoftmaxForward(cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                        &cuda_->one, input_desc_, input->cuda(),
                        &cuda_->zero, output_desc_, output_->cuda());
    return output_;
}

void Softmax::bwd_initialize(Tensor<float> *target) {
    if (grad_input_ == nullptr || batch_size_ != target->n()) {
        if (grad_input_ == nullptr)
            grad_input_ = new Tensor<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
}

Tensor<float> *Softmax::backward(Tensor<float> *target) {
    // set gradient input as predict
    cudaMemcpyAsync(grad_input_->cuda(),
                    output_->cuda(), output_->buf_size(),
                    cudaMemcpyDeviceToDevice);
    // set gradient input = predict - target
    cublasSaxpy(cuda_->cublas(), target->len(),
                &cuda_->minus_one, target->cuda(), 1,
                grad_input_->cuda(), 1);
    // Normalize the gradient output by the batch size
    int grad_output_size = target->n() * target->c() * target->h() * target->w();
    float scale = 1.f / static_cast<float>(target->n());
    cublasSscal(cuda_->cublas(), grad_output_size, &scale, grad_input_->cuda(), 1);
    return grad_input_;
}

float Softmax::get_loss(Tensor<float> *target) {
    return loss_.loss(output_, target);
}

int Softmax::get_accuracy(Tensor<float> *target) {
    int batch_size = output_->n();
    int output_size = output_->size();

    assert(batch_size == target->n());
    assert(output_size == target->size());

    float *h_output, *h_target;
    int idx_output, idx_target;
    int hit_count = 0;

    // get predicts and targets
    h_output = output_->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++) {
        idx_output = 0;
        idx_target = 0;

        for (int i = 1; i < 10; i++) {
            if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
                idx_output = i;
            if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
                idx_target = i;
        }

        // std::cout << "predict: " << idx_output << " target: " << idx_target << std::endl;

        if (idx_output == idx_target)
            hit_count++;
    }

    return hit_count;
}

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/

