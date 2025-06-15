#include "WaveletNoise.h"
#include "PerlinNoise.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

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
    std::cout << "Generated Wavelet 2D Octave " << octave << " noise: " << outputFile << std::endl;
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
    std::cout << "Generated Wavelet 3D Sliced Octave " << octave << " noise: " << outputFile << std::endl;
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
    std::cout << "Generated Wavelet 3D Projected Octave " << octave << " noise: " << outputFile << std::endl;
}

// ===== 新增 Perlin Noise 生成函式 =====
void generatePerlinNoise2D(int imageSize, int octave, const std::string& outputFile, const PerlinNoise& perlin) {
    std::vector<float> image(imageSize * imageSize);
    const float base_range = 4.0f; // 與 Wavelet Noise 使用相同的範圍以進行公平比較
    const float octave_scale = std::pow(2.0f, octave);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            image[y * imageSize + x] = static_cast<float>(perlin.noise(u * octave_scale, v * octave_scale));
        }
    }
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    std::cout << "Generated Perlin 2D Octave " << octave << " noise: " << outputFile << std::endl;
}

void generatePerlinNoise3DSliced(int imageSize, int octave, const std::string& outputFile, const PerlinNoise& perlin) {
    std::vector<float> image(imageSize * imageSize);
    const float base_range = 4.0f;
    const float octave_scale = std::pow(2.0f, octave);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            image[y * imageSize + x] = static_cast<float>(perlin.noise(u * octave_scale, v * octave_scale, 1.0f * octave_scale));
        }
    }
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    std::cout << "Generated Perlin 3D Sliced Octave " << octave << " noise: " << outputFile << std::endl;
}

int main() {
    std::cout << "=== Wavelet & Perlin Noise Comparison Generation ===" << std::endl;
    
    const int IMAGE_SIZE = 256;
    const int TILE_SIZE = 128;
    const unsigned int SEED = 12345;
    
    // 初始化 Wavelet Noise
    std::cout << "\n--- Initializing Wavelet Noise ---" << std::endl;
    WaveletNoise noise2D(TILE_SIZE, SEED);
    noise2D.generateNoiseTile2D();
    WaveletNoise noise3D(TILE_SIZE, SEED);
    noise3D.generateNoiseTile3D();

    // 初始化 Perlin Noise (使用相同種子)
    std::cout << "\n--- Initializing Perlin Noise ---" << std::endl;
    PerlinNoise perlin(SEED);

    // 我們將生成多個頻帶 (octaves) 用於 Figure 8 和 9 的分析
    // Octave 4 是一個很好的單一頻帶範例
    // Octaves 3, 4, 5 將用於多頻帶分析
    std::vector<int> octaves_to_generate = {3, 4, 5};

    for (int octave : octaves_to_generate) {
        std::cout << "\n--- Generating Data for Octave " << octave << " ---" << std::endl;
        std::string o_str = std::to_string(octave);

        // Wavelet Noise
        generate2DOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_2D_octave_" + o_str + ".raw", noise2D);
        generate3DSlicedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dsliced_octave_" + o_str + ".raw", noise3D);
        generate3DProjectedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dprojected_octave_" + o_str + ".raw", noise3D);

        // Perlin Noise
        generatePerlinNoise2D(IMAGE_SIZE, octave, "perlin_noise_2D_octave_" + o_str + ".raw", perlin);
        generatePerlinNoise3DSliced(IMAGE_SIZE, octave, "perlin_noise_3Dsliced_octave_" + o_str + ".raw", perlin);
    }

    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "Generated files include both Wavelet and Perlin noise for comparison." << std::endl;
    std::cout << "Please run 'python3 analyze.py' to visualize the results and compare with the paper." << std::endl;
    
    return 0;
}