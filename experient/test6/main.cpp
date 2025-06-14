#include "WaveletNoise.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

// Single band noise evaluation
float evaluateSingleBand(WaveletNoise& noise, float x, float y, float z, int band) {
    float scale = std::pow(2.0f, band);
    float p[3] = {2.0f * x * scale, 2.0f * y * scale, 2.0f * z * scale};
    float result = noise.evaluate3D(p);
    
    // Normalize by 3D noise variance from paper (0.210)
    return result / std::sqrt(0.210f);
}

// Generate single band noise to demonstrate band-limited property
void generateSingleBandNoise(int imageSize, int band, const std::string& outputFile) {
    // Initialize noise generator
    const int TILE_SIZE = 128;  // Even tile size as required
    WaveletNoise noise(TILE_SIZE, 12345);
    
    std::cout << "Generating noise tile..." << std::endl;
    noise.generateNoiseTile3D();
    
    // Get noise coefficients and print statistics
    const auto& coeffs = noise.getNoiseCoefficients();
    DataStats tileStats = noise.calculateStats(coeffs, "Noise Tile Coefficients");
    
    std::cout << "\nGenerating single band noise at band " << band << std::endl;
    
    // Generate 2D image
    std::vector<float> image(imageSize * imageSize);
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    
    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // Normalized coordinates
            float u = static_cast<float>(x) / imageSize;
            float v = static_cast<float>(y) / imageSize;
            
            // Scale to show appropriate detail for the band
            // Higher bands (more negative) show finer detail
            float scale = 4.0f; // Show 4x4 tile repetition
            float tx = u * scale;
            float ty = v * scale;
            float tz = 0.5f;
            
            // Evaluate single band
            float value = evaluateSingleBand(noise, tx, ty, tz, band);
            
            image[y * imageSize + x] = value;
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        }
    }
    
    // Print image statistics
    DataStats imgStats = noise.calculateStats(image, "Generated Single Band Image");
    std::cout << "Value range: [" << minVal << ", " << maxVal << "]" << std::endl;
    
    // Normalize to [0, 1] for output
    float range = maxVal - minVal;
    if (range > 0) {
        for (float& val : image) {
            val = (val - minVal) / range;
        }
    }
    
    // Save as raw float data
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Cannot open output file " << outputFile << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(image.data()), 
                  image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Saved output to " << outputFile << std::endl;
}

// Compare with multi-band for reference
void generateMultiBandComparison(int imageSize) {
    const int TILE_SIZE = 128;
    WaveletNoise noise(TILE_SIZE, 12345);
    noise.generateNoiseTile3D();
    
    // Generate with 3 bands for comparison
    std::vector<float> image(imageSize * imageSize);
    const int NUM_BANDS = 3;
    const int FIRST_BAND = -2;
    
    std::cout << "\nGenerating 3-band comparison image..." << std::endl;
    
    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = static_cast<float>(x) / imageSize * 4.0f;
            float v = static_cast<float>(y) / imageSize * 4.0f;
            
            float value = 0.0f;
            float totalWeight = 0.0f;
            
            for (int b = 0; b < NUM_BANDS; ++b) {
                float weight = std::pow(0.5f, b); // Each band half amplitude of previous
                value += weight * evaluateSingleBand(noise, u, v, 0.5f, FIRST_BAND + b);
                totalWeight += weight * weight;
            }
            
            // Normalize by total variance
            if (totalWeight > 0) {
                value /= std::sqrt(totalWeight);
            }
            
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize and save
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    
    if (range > 0) {
        for (float& val : image) {
            val = (val - minVal) / range;
        }
    }
    
    std::ofstream outFile("wavelet_noise_multiband_3.raw", std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), 
                  image.size() * sizeof(float));
    outFile.close();
    std::cout << "Saved 3-band comparison to wavelet_noise_multiband_3.raw" << std::endl;
}

int main() {
    std::cout << "=== Wavelet Noise Band-Limited Demonstration ===" << std::endl;
    
    const int IMAGE_SIZE = 512;
    
    // Generate single band at different frequencies to show band-limited property
    std::cout << "\nGenerating single band noise at different frequencies..." << std::endl;
    
    // Band -1: Finest detail that's just representable
    generateSingleBandNoise(IMAGE_SIZE, -1, "wavelet_noise_band_-1.raw");
    
    // Band -2: One octave coarser
    generateSingleBandNoise(IMAGE_SIZE, -2, "wavelet_noise_band_-2.raw");
    
    // Band -3: Two octaves coarser
    generateSingleBandNoise(IMAGE_SIZE, -3, "wavelet_noise_band_-3.raw");
    
    // Also generate a multi-band for comparison
    generateMultiBandComparison(IMAGE_SIZE);
    
    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "Generated files:" << std::endl;
    std::cout << "  - wavelet_noise_band_-1.raw: Finest band (highest frequency)" << std::endl;
    std::cout << "  - wavelet_noise_band_-2.raw: Medium band" << std::endl;
    std::cout << "  - wavelet_noise_band_-3.raw: Coarse band (lowest frequency)" << std::endl;
    std::cout << "  - wavelet_noise_multiband_3.raw: 3-band combination for comparison" << std::endl;
    std::cout << "\nUse the Python analysis script to visualize the band-limited property!" << std::endl;
    
    return 0;
}