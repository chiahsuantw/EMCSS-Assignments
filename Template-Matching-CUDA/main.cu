#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cfloat>
#include <cstdlib>

#define KERNEL_PATH "dataset/0/S5_6_6.txt"
#define MATRIX_PATH "dataset/0/T5_2500_750.txt"
#define MATRIX_HEIGHT 2500
#define MATRIX_WIDTH 750
#define KERNEL_HEIGHT 6
#define KERNEL_WIDTH 6
#define FUNCTION "PCC" // PCC or SSD

#define SUM_HEIGHT (MATRIX_HEIGHT - KERNEL_HEIGHT / 2 * 2)
#define SUM_WIDTH (MATRIX_WIDTH - KERNEL_WIDTH / 2 * 2)
#define BLOCK_SIZE 16

// Calculate the Pearson Correlation Coefficient
__device__ __host__ float PCC(const float *matX, const float *matY) {
    // Calculate the means of X and Y
    float meanX = 0, meanY = 0;
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            meanX += matX[i * KERNEL_HEIGHT + j];
            meanY += matY[i * KERNEL_HEIGHT + j];
        }
    }
    meanX /= (KERNEL_WIDTH * KERNEL_HEIGHT);
    meanY /= (KERNEL_WIDTH * KERNEL_HEIGHT);

    // Calculate the numerator and denominators of the correlation coefficient
    float numerator = 0, denominatorX = 0, denominatorY = 0;
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            numerator += (matX[i * KERNEL_HEIGHT + j] - meanX) * (matY[i * KERNEL_HEIGHT + j] - meanY);
            denominatorX += (matX[i * KERNEL_HEIGHT + j] - meanX) * (matX[i * KERNEL_HEIGHT + j] - meanX);
            denominatorY += (matY[i * KERNEL_HEIGHT + j] - meanY) * (matY[i * KERNEL_HEIGHT + j] - meanY);
        }
    }

    // Calculate the correlation coefficient
    float denominator = std::sqrt(denominatorX) * std::sqrt(denominatorY);
    if (denominator == 0) return 0;
    return numerator / denominator;
}

// Calculate the Sum of Square Difference
__device__ __host__ float SSD(const float *matX, const float *matY) {
    float sum = 0;
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            sum += (matX[i * KERNEL_HEIGHT + j] - matY[i * KERNEL_HEIGHT + j]) *
                   (matX[i * KERNEL_HEIGHT + j] - matY[i * KERNEL_HEIGHT + j]);
        }
    }
    return sum;
}

// Print the matrix
__device__ __host__ void printMatrix(float *matrix, int width, int height) {
    if (matrix == nullptr) {
        printf("Fail to print the matrix");
        return;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

// Kernel function to calculate the Pearson Correlation Coefficient
__global__ void CalculatePCC(const float *matrix, size_t ldm, const float *kernel, size_t ldk, float *sum, size_t lds) {
    __shared__ float filter[KERNEL_WIDTH * KERNEL_HEIGHT];
    __shared__ float tile[(BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) * (BLOCK_SIZE + KERNEL_HEIGHT / 2 * 2)];

    // Copy filter to shared memory
    if (threadIdx.x < KERNEL_HEIGHT && threadIdx.y < KERNEL_WIDTH) {
        filter[threadIdx.x * KERNEL_WIDTH + threadIdx.y] = kernel[threadIdx.x * ldk + threadIdx.y];
    }

    // Copy tile to shared memory
    for (int i = (int) threadIdx.x; i < (BLOCK_SIZE + KERNEL_HEIGHT / 2 * 2) && i < MATRIX_HEIGHT; i += BLOCK_SIZE) {
        for (int j = (int) threadIdx.y; j < (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) && j < MATRIX_WIDTH; j += BLOCK_SIZE) {
            tile[i * (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) + j] =
                    matrix[(blockIdx.x * BLOCK_SIZE + i) * ldm + (blockIdx.y * BLOCK_SIZE + j)];
        }
    }

    __syncthreads(); // Wait for copying data

    // Extract current kernel-size image from the tile
    float image[KERNEL_WIDTH * KERNEL_HEIGHT];
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            image[i * KERNEL_WIDTH + j] = tile[(threadIdx.x + i) * (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) +
                                               (threadIdx.y + j)];
        }
    }

    // Calculate the result
    if (blockIdx.x * BLOCK_SIZE + threadIdx.x < SUM_HEIGHT && blockIdx.y * BLOCK_SIZE + threadIdx.y < SUM_WIDTH) {
        sum[(blockIdx.x * BLOCK_SIZE + threadIdx.x) * lds +
            (blockIdx.y * BLOCK_SIZE + threadIdx.y)] = PCC(image, filter);
    }
}

// Kernel function to calculate the Sum of Square Difference
__global__ void CalculateSSD(const float *matrix, size_t ldm, const float *kernel, size_t ldk, float *sum, size_t lds) {
    __shared__ float filter[KERNEL_WIDTH * KERNEL_HEIGHT];
    __shared__ float tile[(BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) * (BLOCK_SIZE + KERNEL_HEIGHT / 2 * 2)];

    // Copy filter to shared memory
    if (threadIdx.x < KERNEL_HEIGHT && threadIdx.y < KERNEL_WIDTH) {
        filter[threadIdx.x * KERNEL_WIDTH + threadIdx.y] = kernel[threadIdx.x * ldk + threadIdx.y];
    }

    // Copy tile to shared memory
    for (int i = (int) threadIdx.x; i < (BLOCK_SIZE + KERNEL_HEIGHT / 2 * 2) && i < MATRIX_HEIGHT; i += BLOCK_SIZE) {
        for (int j = (int) threadIdx.y; j < (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) && j < MATRIX_WIDTH; j += BLOCK_SIZE) {
            tile[i * (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) + j] =
                    matrix[(blockIdx.x * BLOCK_SIZE + i) * ldm + (blockIdx.y * BLOCK_SIZE + j)];
        }
    }

    __syncthreads(); // Wait for copying data

    // Extract current kernel-size image from the tile
    float image[KERNEL_WIDTH * KERNEL_HEIGHT];
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            image[i * KERNEL_WIDTH + j] = tile[(threadIdx.x + i) * (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) +
                                               (threadIdx.y + j)];
        }
    }

    // Calculate the result
    if (blockIdx.x * BLOCK_SIZE + threadIdx.x < SUM_HEIGHT && blockIdx.y * BLOCK_SIZE + threadIdx.y < SUM_WIDTH) {
        sum[(blockIdx.x * BLOCK_SIZE + threadIdx.x) * lds +
            (blockIdx.y * BLOCK_SIZE + threadIdx.y)] = SSD(image, filter);
    }

    // Print kernel and block at block(0, 0)
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printMatrix(filter, KERNEL_WIDTH, KERNEL_HEIGHT);
    //     printMatrix(tile, (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2), (BLOCK_SIZE + KERNEL_HEIGHT / 2 * 2));
    //     printMatrix(image, KERNEL_WIDTH, KERNEL_HEIGHT);
    // }
}

// Read matrix from the specified text file
float *readMatrixFromFile(const std::string &filepath, int width, int height) {
    std::fstream ifs;
    ifs.open(filepath, std::ios::in);

    if (ifs.fail()) {
        std::cout << "Fail to read the file" << std::endl;
        exit(EXIT_FAILURE);
        return nullptr;
    }

    auto *matrix = new float[width * height]{0};
    int rowCounter = 0, elementCounter = 0;
    std::string line, element;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        int columnCounter = 0;
        while (std::getline(ss, element, ',')) {
            matrix[rowCounter * width + columnCounter] = std::stof(element);
            columnCounter++;
            elementCounter++;
        }
        rowCounter++;
    }
    ifs.close();

    return matrix;
}

// Write matrix to the specified text file
void writeMatrixToFile(const std::string &filepath, const float *matrix, int width, int height) {
    if (matrix == nullptr) return;
    std::fstream ofs;
    ofs.open(filepath, std::ios::out);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (j != 0) ofs << ",";
            ofs << matrix[i * width + j];
        }
        ofs << std::endl;
    }
    ofs.close();
}

// Calculate using CPU
float *calculateFuncUsingCPU(const float *matrix, const float *kernel, float (*func)(const float *, const float *)) {
    auto *results = new float[SUM_WIDTH * SUM_HEIGHT]{0};
    for (int i = 0; i < SUM_HEIGHT; i++) {
        for (int j = 0; j < SUM_WIDTH; j++) {
            auto *image = new float[KERNEL_WIDTH * KERNEL_HEIGHT]{0};
            for (int m = 0; m < KERNEL_HEIGHT; m++) {
                for (int n = 0; n < KERNEL_WIDTH; n++) {
                    image[m * KERNEL_WIDTH + n] = matrix[(i + m) * MATRIX_WIDTH + (j + n)];
                }
            }
            results[i * SUM_WIDTH + j] = func(image, kernel);
        }
    }
    return results;
}

// Compare two files and print the line numbers of different line
void compareFiles(const std::string &filepath1, const std::string &filepath2) {
    std::fstream file1, file2;
    file1.open(filepath1, std::ios::in);
    file2.open(filepath2, std::ios::in);
    int index = 0;
    int counter = 0;
    while (!file1.eof() || !file2.eof()) {
        std::string a, b;
        getline(file1, a);
        getline(file2, b);
        if (a != b) {
            std::cout << index + 1 << std::endl;
            counter++;
        }
        index++;
    }
    std::cout << std::endl << "Error: " << counter << std::endl;
}

// Find best match (max for PCC, min for SSD)
std::vector<std::pair<int, int>> findBestMatch(const float *sum, int width, int height, bool findMax) {
    std::vector<std::pair<int, int>> result;
    float max = 0, min = FLT_MAX;
    if (findMax) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (sum[i * width + j] > max) {
                    max = sum[i * width + j];
                    result.clear();
                    std::pair<int, int> p(i, j);
                    result.push_back(p);
                    // std::cout << "(" << i << ", " << j << ") = " << sum[i * width + j] << std::endl;
                } else if (sum[i * width + j] == max) {
                    std::pair<int, int> p(i, j);
                    result.push_back(p);
                }
            }
        }
    } else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (sum[i * width + j] < min) {
                    min = sum[i * width + j];
                    result.clear();
                    std::pair<int, int> p(i, j);
                    result.push_back(p);
                    // std::cout << "(" << i << ", " << j << ") = " << sum[i * width + j] << std::endl;
                } else if (sum[i * width + j] == min) {
                    std::pair<int, int> p(i, j);
                    result.push_back(p);
                }
            }
        }
    }
    return result;
}

// Show the kernel-size image in matrix at specified row and column
float *showImageInMatrix(const float *matrix, int row, int column) {
    auto *result = new float[KERNEL_WIDTH * KERNEL_HEIGHT]{0};
    if (matrix == nullptr) return result;
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            result[i * KERNEL_WIDTH + j] = matrix[(i + row) * MATRIX_WIDTH + (j + column)];
        }
    }
    return result;
}

int main() {
    // Read the input data
    std::cout << "Start reading data" << std::endl;
    float *matrix = readMatrixFromFile(MATRIX_PATH, MATRIX_WIDTH, MATRIX_HEIGHT);
    float *kernel = readMatrixFromFile(KERNEL_PATH, KERNEL_WIDTH, KERNEL_HEIGHT);
    std::cout << "Finish reading data" << std::endl;

    // Start cuda event recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);

    // Allocate memory spaces for GPU
    float *matrix_d, *kernel_d, *sum_d;
    size_t mPitch, kPitch, sPitch; // Pitches for kernel, matrix, sum
    cudaMallocPitch(&matrix_d, &mPitch, sizeof(float) * MATRIX_WIDTH, MATRIX_HEIGHT);
    cudaMallocPitch(&kernel_d, &kPitch, sizeof(float) * KERNEL_WIDTH, KERNEL_HEIGHT);
    cudaMallocPitch(&sum_d, &sPitch, sizeof(float) * SUM_WIDTH, SUM_HEIGHT);

    // Copy data to the device
    cudaMemcpy2D(matrix_d, mPitch, matrix, sizeof(float) * MATRIX_WIDTH,
                 sizeof(float) * MATRIX_WIDTH, MATRIX_HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy2D(kernel_d, kPitch, kernel, sizeof(float) * KERNEL_WIDTH,
                 sizeof(float) * KERNEL_WIDTH, KERNEL_HEIGHT, cudaMemcpyHostToDevice);

    // Launch the kernel
    std::cout << "Launching " << FUNCTION << " kernel..." << std::endl;
    dim3 blockCount(ceil((float) SUM_HEIGHT / BLOCK_SIZE), ceil((float) SUM_WIDTH / BLOCK_SIZE));
    dim3 threadCount(BLOCK_SIZE, BLOCK_SIZE);
    if (FUNCTION == "PCC") {
        CalculatePCC<<<blockCount, threadCount>>>(
                matrix_d, mPitch / sizeof(float),
                kernel_d, kPitch / sizeof(float),
                sum_d, sPitch / sizeof(float)
        );
    } else {
        CalculateSSD<<<blockCount, threadCount>>>(
                matrix_d, mPitch / sizeof(float),
                kernel_d, kPitch / sizeof(float),
                sum_d, sPitch / sizeof(float)
        );
    }

    // Copy data back to host
    auto *sum = new float[SUM_WIDTH * SUM_HEIGHT]{0};
    cudaMemcpy2D(sum, sizeof(float) * SUM_WIDTH, sum_d, sPitch,
                 sizeof(float) * SUM_WIDTH, SUM_HEIGHT, cudaMemcpyDeviceToHost);

    // Stop cuda event recording
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Finish computation\nElapsed time: " << elapsedTime / 1000 << "s" << std::endl;

    // Find best matches
    std::cout << "Finding best matches..." << std::endl;
    auto bestMatches = findBestMatch(sum, SUM_WIDTH, SUM_HEIGHT, FUNCTION == "PCC");
    for (auto i = bestMatches.begin(); i != bestMatches.end(); i++) {
        std::cout << "(" << i->first << ", " << i->second << ")" << std::endl;
    }

    // Release the allocated spaces
    cudaFree(matrix_d);
    cudaFree(kernel_d);
    cudaFree(sum_d);
    delete[] matrix;
    delete[] kernel;
    delete[] sum;

    return 0;
}
