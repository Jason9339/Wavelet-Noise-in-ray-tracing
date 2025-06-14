#include "sparse_convolution_noise.h"
#include <cmath>

SparseConvolutionNoise::SparseConvolutionNoise(float density, float kernelRadius) 
    : density(density), kernelRadius(kernelRadius) {
    NoiseUtils::log("SparseConvolutionNoise: Initializing with density " + 
                   std::to_string(density) + " and kernel radius " + 
                   std::to_string(kernelRadius));
}

unsigned int SparseConvolutionNoise::hash(int x, int y) {
    unsigned int h = 0;
    h ^= x * 73856093;
    h ^= y * 19349663;
    return h;
}

std::vector<SparseConvolutionNoise::ImpulsePoint> 
SparseConvolutionNoise::generateImpulses(int cellX, int cellY) {
    std::vector<ImpulsePoint> impulses;
    
    unsigned int h = hash(cellX, cellY);
    std::mt19937 gen(h);
    std::poisson_distribution<int> poisson(density);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    
    int numImpulses = poisson(gen);
    
    for (int i = 0; i < numImpulses; i++) {
        ImpulsePoint p;
        p.x = cellX + uniform(gen);
        p.y = cellY + uniform(gen);
        p.amplitude = normal(gen);
        impulses.push_back(p);
    }
    
    return impulses;
}

float SparseConvolutionNoise::kernel(float distance) {
    if (distance >= kernelRadius) return 0.0f;
    
    // Gaussian kernel
    float t = distance / kernelRadius;
    return std::exp(-4.0f * t * t);
}

float SparseConvolutionNoise::evaluate2D(float x, float y) {
    float cellSize = kernelRadius * 2;
    int minCellX = static_cast<int>(std::floor((x - kernelRadius) / cellSize));
    int maxCellX = static_cast<int>(std::floor((x + kernelRadius) / cellSize));
    int minCellY = static_cast<int>(std::floor((y - kernelRadius) / cellSize));
    int maxCellY = static_cast<int>(std::floor((y + kernelRadius) / cellSize));
    
    float sum = 0.0f;
    
    for (int cellY = minCellY; cellY <= maxCellY; cellY++) {
        for (int cellX = minCellX; cellX <= maxCellX; cellX++) {
            auto impulses = generateImpulses(cellX, cellY);
            
            for (const auto& impulse : impulses) {
                float dx = x - impulse.x;
                float dy = y - impulse.y;
                float distance = std::sqrt(dx * dx + dy * dy);
                sum += impulse.amplitude * kernel(distance);
            }
        }
    }
    
    return sum;
}