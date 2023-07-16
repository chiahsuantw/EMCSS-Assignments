#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cfloat>
#include <pthread.h>
#include <sys/time.h>

#define MAX_THREAD_NUM 16

enum Method {
    PCC,
    SSD
};

struct MatrixDimension {
    int width;
    int height;
};

struct ThreadData {
    int rowStart;
    int rowEnd;
    int colStart;
    int colEnd;
    Method method;
};

// Global variables
MatrixDimension kernelDim, sourceDim, resultDim;
float *kernel, *source, *result;
// float max = 0, min = FLT_MAX;
// std::vector<std::pair<int, int>> bestMatches;
// pthread_mutex_t mutex; // Mutex for accessing bestMatches vector

// Calculate the Pearson Correlation Coefficient
float calculatePCC(const float *matX, const float *matY, MatrixDimension dim) {
    // Calculate the means of X and Y
    float meanX = 0, meanY = 0;
    for (int i = 0; i < dim.height; i++) {
        for (int j = 0; j < dim.width; j++) {
            meanX += matX[i * dim.height + j];
            meanY += matY[i * dim.height + j];
        }
    }
    meanX /= (float) (dim.width * dim.height);
    meanY /= (float) (dim.width * dim.height);

    // Calculate the numerator and denominators of the correlation coefficient
    float numerator = 0, denominatorX = 0, denominatorY = 0;
    for (int i = 0; i < dim.height; i++) {
        for (int j = 0; j < dim.width; j++) {
            numerator += (matX[i * dim.height + j] - meanX) * (matY[i * dim.height + j] - meanY);
            denominatorX += (matX[i * dim.height + j] - meanX) * (matX[i * dim.height + j] - meanX);
            denominatorY += (matY[i * dim.height + j] - meanY) * (matY[i * dim.height + j] - meanY);
        }
    }

    // Calculate the correlation coefficient
    float denominator = std::sqrt(denominatorX) * std::sqrt(denominatorY);
    if (denominator == 0) return 0;
    return numerator / denominator;
}

// Calculate the Sum of Square Difference
float calculateSSD(const float *matX, const float *matY, MatrixDimension dim) {
    float sum = 0;
    for (int i = 0; i < dim.height; i++) {
        for (int j = 0; j < dim.width; j++) {
            sum += (matX[i * dim.height + j] - matY[i * dim.height + j]) *
                   (matX[i * dim.height + j] - matY[i * dim.height + j]);
        }
    }
    return sum;
}

// Thread function
void *run(void *arg) {
    auto data = (ThreadData *) arg;
    for (int i = data->rowStart; i < data->rowEnd; i++) {
        for (int j = data->colStart; j < data->colEnd; j++) {
            // Extract the image to be compared
            float image[kernelDim.width * kernelDim.height];
            for (int m = 0; m < kernelDim.height; m++) {
                for (int n = 0; n < kernelDim.width; n++) {
                    image[m * kernelDim.width + n] = source[(i + m) * sourceDim.width + (j + n)];
                }
            }

            // Calculate the result based on the specified method
            if (data->method == PCC)
                result[i * resultDim.width + j] = calculatePCC(kernel, image, kernelDim);
            else if (data->method == SSD)
                result[i * resultDim.width + j] = calculateSSD(kernel, image, kernelDim);

            // Update the current best matches
            // pthread_mutex_lock(&mutex);
            // if (data->method == PCC) {
            //     if (result[i * resultDim.width + j] > max) {
            //         max = result[i * resultDim.width + j];
            //         bestMatches.clear();
            //         std::pair<int, int> p(i, j);
            //         bestMatches.push_back(p);
            //         // std::cout << "(" << i << ", " << j << ") = " << sum[i * width + j] << std::endl;
            //     } else if (result[i * resultDim.width + j] == max) {
            //         std::pair<int, int> p(i, j);
            //         bestMatches.push_back(p);
            //     }
            // } else if (data->method == SSD) {
            //     if (result[i * resultDim.width + j] < min) {
            //         min = result[i * resultDim.width + j];
            //         bestMatches.clear();
            //         std::pair<int, int> p(i, j);
            //         bestMatches.push_back(p);
            //         // std::cout << "(" << i << ", " << j << ") = " << sum[i * width + j] << std::endl;
            //     } else if (result[i * resultDim.width + j] == min) {
            //         std::pair<int, int> p(i, j);
            //         bestMatches.push_back(p);
            //     }
            // }
            // pthread_mutex_unlock(&mutex);
        }
    }
    pthread_exit(nullptr);
    return nullptr;
}

MatrixDimension getMatrixDimensionFromFile(const std::string &filepath) {
    // filename example: S1_3_3.txt
    std::stringstream ss(filepath);
    std::string temp, width, height;
    std::getline(ss, temp, '_');
    std::getline(ss, height, '_');
    std::getline(ss, width, '_');
    return {.width=std::stoi(width), .height=std::stoi(height)};
}

// Read matrix from the specified text file
float *readMatrixFromFile(const std::string &filepath, MatrixDimension dim) {
    std::fstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail()) return nullptr;

    auto *matrix = new float[dim.width * dim.height]{0};
    int rowCounter = 0, elementCounter = 0;
    std::string line, element;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        int columnCounter = 0;
        while (std::getline(ss, element, ',')) {
            matrix[rowCounter * dim.width + columnCounter] = std::stof(element);
            columnCounter++;
            elementCounter++;
        }
        rowCounter++;
    }
    ifs.close();

    return matrix;
}

// Find the best matches
std::vector<std::pair<int, int>> findBestMatches(const float *matrix, MatrixDimension dim, Method method) {
    std::vector<std::pair<int, int>> matches;
    float _max = 0, _min = FLT_MAX;
    for (int i = 0; i < dim.height; i++) {
        for (int j = 0; j < dim.width; j++) {
            if (method == PCC) {
                if (matrix[i * dim.width + j] > _max) {
                    _max = matrix[i * dim.width + j];
                    matches.clear();
                    std::pair<int, int> p(i, j);
                    matches.push_back(p);
                } else if (matrix[i * dim.width + j] == _max) {
                    std::pair<int, int> p(i, j);
                    matches.push_back(p);
                }
            } else if (method == SSD) {
                if (matrix[i * dim.width + j] < _min) {
                    _min = matrix[i * dim.width + j];
                    matches.clear();
                    std::pair<int, int> p(i, j);
                    matches.push_back(p);
                } else if (matrix[i * dim.width + j] == _min) {
                    std::pair<int, int> p(i, j);
                    matches.push_back(p);
                }
            }
        }
    }
    return matches;
}

// Main function
// argv[1]: <kernel filepath>
// argv[2]: <source filepath>
// argv[3]: <method> (PCC/SSD)
int main(int argc, char *argv[]) {
    // Reading data
    if (argc <= 3) {
        std::cout << "Arguments missing" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[Kernel] " << argv[1] << "\n[Matrix] " << argv[2] << std::endl;
    kernelDim = getMatrixDimensionFromFile(argv[1]);
    sourceDim = getMatrixDimensionFromFile(argv[2]);
    resultDim = MatrixDimension{
            .width=sourceDim.width - kernelDim.width + 1,
            .height=sourceDim.height - kernelDim.height + 1
    };
    kernel = readMatrixFromFile(argv[1], kernelDim);
    source = readMatrixFromFile(argv[2], sourceDim);
    result = new float[resultDim.width * resultDim.height]{0};
    if (kernel == nullptr or source == nullptr) {
        std::cout << "Fail to read the file" << std::endl;
        return EXIT_FAILURE;
    }

    Method method;
    if (std::string(argv[3]) == std::string("PCC")) {
        method = PCC;
    } else if (std::string(argv[3]) == std::string("SSD")) {
        method = SSD;
    } else {
        std::cout << "No such method" << std::endl;
        return EXIT_FAILURE;
    }

    timeval tv1{}, tv2{};
    gettimeofday(&tv1, nullptr);
    // pthread_mutex_init(&mutex, nullptr);

    // Create threads
    int createdThreadCount = std::min(MAX_THREAD_NUM, resultDim.height);
    int rowsPerThread = ceil((double) resultDim.height / MAX_THREAD_NUM);
    auto *threads = new pthread_t[createdThreadCount];
    auto *params = new ThreadData[createdThreadCount];
    for (int i = 0; i < createdThreadCount; i++) {
        params[i].rowStart = i * rowsPerThread;
        params[i].rowEnd = (i + 1) * rowsPerThread;
        if (params[i].rowStart > resultDim.height) params[i].rowStart = resultDim.height;
        if (params[i].rowEnd > resultDim.height) params[i].rowEnd = resultDim.height;
        params[i].colStart = 0;
        params[i].colEnd = resultDim.width;
        params[i].method = method;
        pthread_create(&threads[i], nullptr, run, &params[i]);
    }

    // Join threads
    for (int i = 0; i < createdThreadCount; i++) {
        pthread_join(threads[i], nullptr);
    }

    // pthread_mutex_destroy(&mutex);
    gettimeofday(&tv2, nullptr);
    double startTime = tv1.tv_sec * 1000000 + tv1.tv_usec;
    double endTime = tv2.tv_sec * 1000000 + tv2.tv_usec;
    std::cout << "Elapsed time: " << (endTime - startTime) / 1000000 << " s" << std::endl;

    // Print the best matched positions
    std::cout << "Best matches:" << std::endl;
    auto bestMatches = findBestMatches(result, resultDim, method);
    for (auto &pos: bestMatches)
        std::cout << "(" << pos.first << ", " << pos.second << ")" << std::endl;

    // Release the resources
    delete[] kernel;
    delete[] source;
    delete[] result;
    return EXIT_SUCCESS;
}
