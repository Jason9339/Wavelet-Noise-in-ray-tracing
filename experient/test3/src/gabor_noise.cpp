#include "gabor_noise.h"
#include <cmath>

GaborNoise::GaborNoise(float kernelRadius, float frequency, float orientation, float bandwidth)
    : kernelRadius(kernelRadius), frequency(frequency), 
      orientation(orientation), bandwidth(bandwidth) {
    NoiseUtils::log("GaborNoise: Initializing with radius " + std::to_string(kernelRadius) +
                   ", frequency " + std::to_string(frequency) +
                   ", orientation " + std::to_string(orientation) +
                   ", bandwidth " + std::to_string(bandwidth));
}

unsigned int GaborNoise::hash(int x, int y) {
    unsigned int h = 0;
    h ^= x * 73856093;
    h ^= y * 19349663;
    return h;
}

std::vector<GaborNoise::GaborKernel> GaborNoise::generateKernels(int cellX, int cellY) {
    std::vector<GaborKernel> kernels;
    
    unsigned int h = hash(cellX, cellY);
    std::mt19937 gen(h);
    std::poisson_distribution<int> poisson(16.0f); // Higher density than sparse convolution
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    int numKernels = poisson(gen);
    
    for (int i = 0; i < numKernels; i++) {
        GaborKernel k;
        k.x = cellX + uniform(gen);
        k.y = cellY + uniform(gen);
        k.weight = normal(gen);
        k.phase = uniform(gen) * 2 * M_PI;
        k.orientation = orientation + normal(gen) * 0.1f; // Small random variation
        kernels.push_back(k);
    }
    
    return kernels;
}

float GaborNoise::evaluateKernel(float x, float y, const GaborKernel& kernel) {
    float dx = x - kernel.x;
    float dy = y - kernel.y;
    
    // Rotate to kernel orientation
    float cosTheta = cos(kernel.orientation);
    float sinTheta = sin(kernel.orientation);
    float rx = dx * cosTheta + dy * sinTheta;
    float ry = -dx * sinTheta + dy * cosTheta;
    
    // Gaussian envelope
    float gaussianTerm = exp(-M_PI * (rx*rx + ry*ry) / (kernelRadius * kernelRadius));
    
    // Cosine wave
    float cosineTerm = cos(2 * M_PI * frequency * rx + kernel.phase);
    
    return kernel.weight * gaussianTerm * cosineTerm;
}

float GaborNoise::evaluate2D(float x, float y) {
    float cellSize = kernelRadius * 3; // Larger cells for Gabor kernels
    int minCellX = static_cast<int>(std::floor((x - kernelRadius) / cellSize));
    int maxCellX = static_cast<int>(std::floor((x + kernelRadius) / cellSize));
    int minCellY = static_cast<int>(std::floor((y - kernelRadius) / cellSize));
    int maxCellY = static_cast<int>(std::floor((y + kernelRadius) / cellSize));
    
    float sum = 0.0f;
    
    // NoiseUtils::log("GaborNoise: Evaluating at (" + std::to_string(x) + ", " + 
    //                std::to_string(y) + ")");
    
    for (int cellY = minCellY; cellY <= maxCellY; cellY++) {
        for (int cellX = minCellX; cellX <= maxCellX; cellX++) {
            auto kernels = generateKernels(cellX, cellY);
            
            for (const auto& kernel : kernels) {
                float dist = std::sqrt((x - kernel.x)*(x - kernel.x) + 
                                     (y - kernel.y)*(y - kernel.y));
                if (dist < kernelRadius) {
                    sum += evaluateKernel(x, y, kernel);
                }
            }
        }
    }
    
    return sum;
}