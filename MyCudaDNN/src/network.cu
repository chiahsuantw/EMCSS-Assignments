#include <iostream>

#include "network.cuh"
#include "utils.cuh"
#include "layer.cuh"

using namespace cudl;

Network::Network() = default;

Network::~Network() {
    // Destroy network
    for (auto layer: layers_)
        delete layer;

    // Terminate CUDA context
    delete cuda_;
}

void Network::addLayer(Layer *layer) {
    layers_.push_back(layer);

    // Tagging layer to stop gradient if it is the first layer
    if (layers_.size() == 1)
        layers_.at(0)->set_gradient_stop();
}

Tensor<float> *Network::forward(Tensor<float> *input) {
    output_ = input;

    for (auto layer: layers_) {
        layer->fwd_initialize(output_);
        output_ = layer->forward(output_);
    }

    return output_;
}

void Network::backward(Tensor<float> *target) {
    Tensor<float> *gradient = target;

    if (phase_ == inference)
        return;

    // Back-propagation
    // Updating weights internally
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
        // Getting back propagation status with gradient size
        (*layer)->bwd_initialize(gradient);
        gradient = (*layer)->backward(gradient);
    }
}

void Network::update(float learning_rate) {
    if (phase_ == inference)
        return;

    for (auto layer: layers_) {
        // If no parameters, then pass
        if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
            layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
            continue;
        layer->update_weights_biases(learning_rate);
    }
}

int Network::writeFile() {
    std::cout << "\nStoring weights to the storage" << std::endl;
    for (auto layer: layers_) {
        int err = layer->save_parameter();

        if (err != 0) {
            std::cout << "-> error code: " << err << std::endl;
            exit(err);
        }
    }
    return 0;
}

int Network::loadPretrain() {
    for (auto layer: layers_) {
        layer->set_load_pretrain();
    }
    return 0;
}

// Initialize cuda resource container
// Register the resource container to all layers
void Network::cuda() {
    cuda_ = new CudaContext();
    std::cout << "\nListing model layers" << std::endl;
    for (auto layer: layers_) {
        std::cout << "CUDA: " << layer->get_name() << std::endl;
        layer->set_cuda_context(cuda_);
    }
}

void Network::train() {
    phase_ = training;
    // Unfreeze all layers
    for (auto layer: layers_) {
        layer->unfreeze();
    }
}

void Network::test() {
    phase_ = inference;
    // Freeze all layers
    for (auto layer: layers_) {
        layer->freeze();
    }
}

int Network::predict(Tensor<float> *input) {
    test();
    forward(input);

    int batch_size = output_->n();
    int output_size = output_->size();

    float *h_output;
    int idx_output;

    if (batch_size != 1) {
        std::cout << "Predict batch size should be 1" << std::endl;
    }

    h_output = output_->to(host);

    for (int b = 0; b < batch_size; b++) {
        idx_output = 0;

        for (int i = 1; i < 10; i++) {
            if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
                idx_output = i;
        }

        std::cout << "predict: " << idx_output << std::endl;
    }

    return idx_output;
}

float Network::loss(Tensor<float> *target) {
    Layer *layer = layers_.back();
    return layer->get_loss(target);
}

int Network::getAccuracy(Tensor<float> *target) {
    Layer *layer = layers_.back();
    return layer->get_accuracy(target);
}