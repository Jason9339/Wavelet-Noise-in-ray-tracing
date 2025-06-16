#include "anisotropic_noise.h"
#include <cmath>

AnisotropicNoise::AnisotropicNoise(int numOrientations, float anisotropy) 
    : numOrientations(numOrientations), anisotropy(anisotropy) {
    NoiseUtils::log("AnisotropicNoise: Initializing with " + 
                   std::to_string(numOrientations) + " orientations and anisotropy " + 
                   std::to_string(anisotropy));
    generateOrientedBands();
}

void AnisotropicNoise::generateOrientedBands() {
    NoiseUtils::log("AnisotropicNoise: Generating oriented bands");
    
    const int bandSize = 128;
    orientedBands.resize(numOrientations);
    
    // Generate base noise
    std::vector<std::vector<float>> baseNoise(bandSize, std::vector<float>(bandSize));
    for (int y = 0; y < bandSize; y++) {
        for (int x = 0; x < bandSize; x++) {
            baseNoise[y][x] = NoiseUtils::random_gaussian();
        }
    }
    
    // Create oriented bands
    for (int i = 0; i < numOrientations; i++) {
        float angle = i * M_PI / numOrientations;
        NoiseUtils::log("AnisotropicNoise: Creating band for angle " + 
                       std::to_string(angle * 180 / M_PI) + " degrees");
        
        auto filter = generateSteerableFilter(angle, 31);
        orientedBands[i] = applyFilter(baseNoise, filter);
    }
    
    NoiseUtils::log("AnisotropicNoise: Oriented bands generation complete");
}

std::vector<std::vector<float>> AnisotropicNoise::generateSteerableFilter(float angle, int size) {
    std::vector<std::vector<float>> filter(size, std::vector<float>(size));
    int center = size / 2;
    float sigma = size / 6.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            
            // Rotate coordinates
            float rx = dx * cos(angle) + dy * sin(angle);
            float ry = -dx * sin(angle) + dy * cos(angle);
            
            // Anisotropic Gaussian
            float gx = exp(-(rx * rx) / (2 * sigma * sigma));
            float gy = exp(-(ry * ry) / (2 * sigma * sigma * anisotropy * anisotropy));
            
            filter[y][x] = gx * gy;
        }
    }
    
    // Normalize
    float sum = 0.0f;
    for (const auto& row : filter) {
        for (float val : row) {
            sum += val;
        }
    }
    
    for (auto& row : filter) {
        for (float& val : row) {
            val /= sum;
        }
    }
    
    return filter;
}

std::vector<std::vector<float>> AnisotropicNoise::applyFilter(
    const std::vector<std::vector<float>>& noise,
    const std::vector<std::vector<float>>& filter) {
    
    int noiseSize = noise.size();
    int filterSize = filter.size();
    int halfFilter = filterSize / 2;
    
    std::vector<std::vector<float>> result(noiseSize, std::vector<float>(noiseSize, 0.0f));
    
    for (int y = 0; y < noiseSize; y++) {
        for (int x = 0; x < noiseSize; x++) {
            float sum = 0.0f;
            
            for (int fy = 0; fy < filterSize; fy++) {
                for (int fx = 0; fx < filterSize; fx++) {
                    int ny = (y + fy - halfFilter + noiseSize) % noiseSize;
                    int nx = (x + fx - halfFilter + noiseSize) % noiseSize;
                    sum += noise[ny][nx] * filter[fy][fx];
                }
            }
            
            result[y][x] = sum;
        }
    }
    
    return result;
}

float AnisotropicNoise::evaluate2D(float x, float y) {
    // Determine which orientations to blend
    float angle = atan2(y, x);
    if (angle < 0) angle += 2 * M_PI;
    
    float orientationAngle = 2 * M_PI / numOrientations;
    int orient1 = static_cast<int>(angle / orientationAngle) % numOrientations;
    int orient2 = (orient1 + 1) % numOrientations;
    
    float blend = (angle - orient1 * orientationAngle) / orientationAngle;
    
    // Sample from oriented bands
    int bandSize = orientedBands[0].size();
    int ix = static_cast<int>(fmod(x * bandSize, bandSize));
    int iy = static_cast<int>(fmod(y * bandSize, bandSize));
    if (ix < 0) ix += bandSize;
    if (iy < 0) iy += bandSize;
    
    float v1 = orientedBands[orient1][iy][ix];
    float v2 = orientedBands[orient2][iy][ix];
    
    return NoiseUtils::lerp(v1, v2, blend);
}