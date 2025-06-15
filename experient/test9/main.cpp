#include "WaveletNoise.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

// This is the practical, floating-point sampling version that allows for frequency scaling.
void generate2DOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    const float base_range = 4.0f; // Controls how many noise features are visible at octave 0
    const float octave_scale = std::pow(2.0f, octave);
    const float inv_stddev_2d = 1.0f / std::sqrt(0.19686f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // Standard floating-point coordinates
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            // Apply the crucial octave scaling
            float p[2] = { u * octave_scale, v * octave_scale };
            
            // The factor of 2.0 is from the paper's N(2x) formulation in the basis functions
            p[0] *= 2.0f;
            p[1] *= 2.0f;

            image[y * imageSize + x] = noise.evaluate2D(p) * inv_stddev_2d;
        }
    }
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    std::cout << "Generated 2D Octave " << octave << " noise: " << outputFile << std::endl;
}

// Practical, floating-point sampling version
void generate3DSlicedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);

    const float base_range = 4.0f;
    const float octave_scale = std::pow(2.0f, octave);
    const float inv_stddev_3d = 1.0f / std::sqrt(0.18402f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            float p[3] = { u * octave_scale, v * octave_scale, 1.0f };
            
            p[0] *= 2.0f;
            p[1] *= 2.0f;
            p[2] *= 2.0f;

            image[y * imageSize + x] = noise.evaluate3D(p) * inv_stddev_3d;
        }
    }

    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    std::cout << "Generated 3D Sliced Octave " << octave << " noise: " << outputFile << std::endl;
}

// Practical, floating-point sampling version
void generate3DProjectedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    const float base_range = 4.0f;
    const float octave_scale = std::pow(2.0f, octave);
    float normal[3] = {0.0f, 0.0f, 1.0f};
    const float inv_stddev_3d_proj = 1.0f / std::sqrt(0.296f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;

            float p[3] = { u * octave_scale, v * octave_scale, 1.0f };
            
            p[0] *= 2.0f;
            p[1] *= 2.0f;
            p[2] *= 2.0f;
            
            image[y * imageSize + x] = noise.evaluate3DProjected(p, normal) * inv_stddev_3d_proj;
        }
    }
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    std::cout << "Generated 3D Projected Octave " << octave << " noise: " << outputFile << std::endl;
}


int main() {
    std::cout << "=== Wavelet Noise - Final Practical Version (Frequency Control Enabled) ===" << std::endl;
    
    const int IMAGE_SIZE = 256;
    const int TILE_SIZE = 128;
    
    std::cout << "\n--- Generating Pure 2D Noise Patterns ---" << std::endl;
    WaveletNoise noise2D(TILE_SIZE, 12345);
    noise2D.generateNoiseTile2D();
    
    std::cout << "\n--- Generating 3D Sliced & Projected Noise Patterns ---" << std::endl;
    WaveletNoise noise3D(TILE_SIZE, 12345);
    noise3D.generateNoiseTile3D();

    for (int octave = 3; octave <= 5; ++octave) {
        std::cout << "\n--- Generating Octave " << octave << " ---" << std::endl;
        generate2DOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_2D_octave_" + std::to_string(octave) + ".raw", noise2D);
        generate3DSlicedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dsliced_octave_" + std::to_string(octave) + ".raw", noise3D);
        generate3DProjectedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dprojected_octave_" + std::to_string(octave) + ".raw", noise3D);
    }

    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "All octaves generated with correct frequency scaling." << std::endl;
    std::cout << "Please run 'analyze.py' to observe the differences between octaves." << std::endl;
    
    return 0;
}