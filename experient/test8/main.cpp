#include "WaveletNoise.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

// Generates a 2D image of pure 2D band-limited noise at a specific octave.
void generate2DOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    float base_range = 4.0f; // Controls how many noise features are visible
    float octave_scale = std::pow(2.0f, octave);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            // Equation (1): N(2^b * x)
            float p[2] = {2.0f * u * octave_scale, 2.0f * v * octave_scale};
            
            // Section 4.2: The variance of 2D noise is ~0.265
            float value = noise.evaluate2D(p) / std::sqrt(0.265f);
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range > 0) for (float& val : image) val = (val - minVal) / range;
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated pure 2D octave " << octave << " noise: " << outputFile << std::endl;
}

// Generates a 2D image by slicing a 3D band-limited noise volume.
void generate3DSlicedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    float base_range = 4.0f;
    float octave_scale = std::pow(2.0f, octave);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            // Slicing 3D noise: use a constant z
            float p[3] = {2.0f * u * octave_scale, 2.0f * v * octave_scale, 1.0f}; 
            
            // Section 4.2: The variance of 3D noise is ~0.210
            float value = noise.evaluate3D(p) / std::sqrt(0.210f);
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range > 0) for (float& val : image) val = (val - minVal) / range;
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated 3D sliced octave " << octave << " noise: " << outputFile << std::endl;
}

int main() {
    std::cout << "=== Wavelet Noise Implementation Verification ===" << std::endl;
    
    const int IMAGE_SIZE = 256;
    const int TILE_SIZE = 128; // Must be even
    
    // --- Pure 2D Noise Generation and Variance Check ---
    std::cout << "\n--- Generating Pure 2D Noise Patterns ---" << std::endl;
    WaveletNoise noise2D(TILE_SIZE, 12345);
    noise2D.generateNoiseTile2D();
    
    std::cout << "Verifying 2D noise coefficient variance (Theoretical ~0.265)..." << std::endl;
    noise2D.calculateStats(noise2D.getNoiseCoefficients(), "2D Noise Tile Coefficients");
    
    for (int octave = 3; octave <= 5; ++octave) {
        generate2DOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_2D_octave_" + std::to_string(octave) + ".raw", noise2D);
    }

    // --- 3D Sliced Noise Generation and Variance Check ---
    std::cout << "\n--- Generating 3D Sliced Noise Patterns ---" << std::endl;
    WaveletNoise noise3D(TILE_SIZE, 12345);
    noise3D.generateNoiseTile3D();

    std::cout << "Verifying 3D noise coefficient variance (Theoretical ~0.210)..." << std::endl;
    noise3D.calculateStats(noise3D.getNoiseCoefficients(), "3D Noise Tile Coefficients");

    for (int octave = 3; octave <= 5; ++octave) {
        generate3DSlicedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dsliced_octave_" + std::to_string(octave) + ".raw", noise3D);
    }

    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "Please run the 'analyze.py' script to visualize the results." << std::endl;
    
    return 0;
}