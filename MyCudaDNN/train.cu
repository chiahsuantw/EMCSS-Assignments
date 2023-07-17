#include <iomanip>

#include "src/mnist.cuh"
#include "src/network.cuh"
#include "src/layer.cuh"

using namespace cudl;

int main() {
    // Configuring training and testing parameters
    int trainBatchSize = 256;
    int trainStepNum = 1600;
    int monitoringStep = 200;
    double learningRate = 0.02f;
    double lrDecay = 0.00005f;
    bool loadPretrain = false;
    bool saveFile = false;
    int testBatchSize = 10;
    int testStepNum = 1000;

    /* ========================= Training ========================= */
    std::cout << "[TRAINING]" << std::endl;

    // Loading training dataset
    MNIST trainDataLoader = MNIST("./dataset");
    trainDataLoader.train(trainBatchSize, true);

    // Initializing model
    Network model;
    model.addLayer(new Dense("Dense 1", 500));
    model.addLayer(new Activation("ReLU", CUDNN_ACTIVATION_RELU));
    model.addLayer(new Dense("Dense 2", 10));
    model.addLayer(new Softmax("Softmax"));
    model.cuda();

    if (loadPretrain)
        model.loadPretrain();

    model.train();

    // Start training
    std::cout << "\nTraining" << std::endl;
    Tensor<float> *trainData = trainDataLoader.getData();
    Tensor<float> *getTarget = trainDataLoader.getTarget();
    trainDataLoader.getBatch();
    int tp_count = 0;
    int step = 0;
    while (step < trainStepNum) {
        // Updating shared buffer contents
        trainData->to(cuda);
        getTarget->to(cuda);

        // Forward
        model.forward(trainData);
        tp_count += model.getAccuracy(getTarget);

        // Back-propagation
        model.backward(getTarget);

        // Updating parameters
        learningRate *= 1.f / (1.f + lrDecay * step);
        model.update((float) learningRate);

        // Fetching the next data
        step = trainDataLoader.next();

        // Calculating Softmax loss
        if (step % monitoringStep == 0) {
            float loss = model.loss(getTarget);
            float accuracy = 100.f * (float) tp_count / (float) monitoringStep / (float) trainBatchSize;
            std::cout << "step: " << std::right << std::setw(4) << step << \
                         " loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                         " accuracy: " << accuracy << "%" << std::endl;
            tp_count = 0;
        }
    }

    // Saving trained parameters
    if (saveFile)
        model.writeFile();

    /* ========================= Testing ========================= */
    std::cout << "\n\n[TESTING]" << std::endl;

    // Loading testing dataset
    MNIST testDataLoader = MNIST("./dataset");
    testDataLoader.test(testBatchSize);

    // Initializing model
    model.test();

    // Start testing
    Tensor<float> *testData = testDataLoader.getData();
    Tensor<float> *testTarget = testDataLoader.getTarget();
    testDataLoader.getBatch();
    tp_count = 0;
    step = 0;
    while (step < testStepNum) {
        // Updating shared buffer contents
        testData->to(cuda);
        testTarget->to(cuda);

        // Forward
        model.forward(testData);
        tp_count += model.getAccuracy(testTarget);

        // Fetching the next data
        step = testDataLoader.next();
    }

    // Calculating loss and accuracy
    float loss = model.loss(testTarget);
    float accuracy = 100.f * (float) tp_count / (float) testStepNum / (float) testBatchSize;
    std::cout << "loss: " << std::setw(4) << loss << " accuracy: " << accuracy << "%" << std::endl;

    /* ========================= Inference ========================= */
    std::cout << "\n\n[INFERENCE]" << std::endl;

    // Loading image
    MNIST dataLoader = MNIST(".");
    dataLoader.loadImage("5.jpg");

    // Predict
    Tensor<float> *image = dataLoader.getData();
    image->to(cuda);
    model.predict(image);

    return 0;
}
