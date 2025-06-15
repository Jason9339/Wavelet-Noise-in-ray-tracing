#include "WaveletNoise.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

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
    
    // Calculate the frequency multiplier for this band
    float bandScale = std::pow(2.0f, band);
    
    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // Map pixel to world coordinates
            // We want to see multiple periods of the noise pattern
            float worldX = static_cast<float>(x) / imageSize * 8.0f;  // Show 8 units
            float worldY = static_cast<float>(y) / imageSize * 8.0f;
            float worldZ = 0.5f;
            
            // Scale by 2^band as per paper's formulation
            // For band -1: scale = 0.5, so we see higher frequency
            // For band -2: scale = 0.25, so we see lower frequency
            // For band -3: scale = 0.125, so we see even lower frequency
            float p[3] = {
                2.0f * worldX * bandScale,
                2.0f * worldY * bandScale,
                2.0f * worldZ * bandScale
            };
            
            // Evaluate noise at this point
            float value = noise.evaluate3D(p);
            
            // Normalize by expected variance (0.210 for 3D noise from paper)
            value /= std::sqrt(0.210f);
            
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

// Generate proper band-limited noise at specific octave
void generateOctaveBandNoise(int imageSize, int octave, const std::string& outputFile) {
    const int TILE_SIZE = 128;
    WaveletNoise noise(TILE_SIZE, 12345);
    noise.generateNoiseTile3D();
    
    std::vector<float> image(imageSize * imageSize);
    
    // 固定一個較小的觀察範圍，比如在圖像上顯示 4 個單位
    float base_range = 4.0f;
    // 頻率縮放因子
    float octave_scale = std::pow(2.0f, octave);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // 基礎採樣點 s = (u,v)
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;

            // 根據 octave 縮放採樣點
            float scaled_u = u * octave_scale;
            float scaled_v = v * octave_scale;
            
            // 論文提到 evaluate3D 的輸入點 p = 2 * s
            float p[3] = {2.0f * scaled_u, 2.0f * scaled_v, 1.0f}; 
            
            float value = noise.evaluate3D(p) / std::sqrt(0.210f);
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    
    if (range > 0) {
        for (float& val : image) {
            val = (val - minVal) / range;
        }
    }
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), 
                  image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated octave " << octave << " noise: " << outputFile << std::endl;
}

// Compare with multi-band for reference
void generateMultiBandComparison(int imageSize) {
    const int TILE_SIZE = 128;
    WaveletNoise noise(TILE_SIZE, 12345);
    noise.generateNoiseTile3D();
    
    std::vector<float> image(imageSize * imageSize);
    
    std::cout << "\nGenerating 3-band comparison image..." << std::endl;
    
    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = static_cast<float>(x) / imageSize * 4.0f;
            float v = static_cast<float>(y) / imageSize * 4.0f;
            
            float value = 0.0f;
            float totalWeight = 0.0f;
            
            // Add 3 octaves
            for (int octave = 0; octave < 3; ++octave) {
                float scale = std::pow(2.0f, octave);
                float weight = std::pow(0.5f, octave);
                
                float p[3] = {2.0f * u * scale, 2.0f * v * scale, 1.0f};
                value += weight * noise.evaluate3D(p);
                totalWeight += weight * weight;
            }
            
            // Normalize by total variance
            value /= std::sqrt(totalWeight * 0.210f);
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

// ===== 新增的函式，用於生成和測試純 2D 噪聲 =====
void generate2DOctaveBandNoise(int imageSize, int octave, const std::string& outputFile) {
    const int TILE_SIZE = 128;
    WaveletNoise noise(TILE_SIZE, 12345);
    
    std::cout << "\nGenerating 2D noise tile for octave " << octave << "..." << std::endl;
    noise.generateNoiseTile2D(); // <--- 呼叫新的 2D 版本
    noise.calculateStats(noise.getNoiseCoefficients(), "2D Noise Tile Coefficients");
    
    std::vector<float> image(imageSize * imageSize);
    
    float base_range = 4.0f;
    float octave_scale = std::pow(2.0f, octave);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            float scaled_u = u * octave_scale;
            float scaled_v = v * octave_scale;
            
            float p[2] = {2.0f * scaled_u, 2.0f * scaled_v}; 
            
            // 呼叫新的 2D 求值函式
            // 根據論文，2D 噪聲的理論方差約為 0.265
            float value = noise.evaluate2D(p) / std::sqrt(0.265f);
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    
    if (range > 0) {
        for (float& val : image) {
            val = (val - minVal) / range;
        }
    }
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), 
                  image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated 2D octave " << octave << " noise: " << outputFile << std::endl;
}

int main() {
    std::cout << "=== Wavelet Noise Band-Limited Demonstration ===" << std::endl;
    
    const int IMAGE_SIZE = 512;
    
    // ===== 測試 3D 切片噪聲 (保留原有測試) =====
    std::cout << "\n--- Generating 3D Sliced Noise Patterns ---" << std::endl;
    generateOctaveBandNoise(IMAGE_SIZE, 3, "wavelet_noise_3Dsliced_octave_3.raw");
    generateOctaveBandNoise(IMAGE_SIZE, 4, "wavelet_noise_3Dsliced_octave_4.raw");
    generateOctaveBandNoise(IMAGE_SIZE, 5, "wavelet_noise_3Dsliced_octave_5.raw");

    // ===== 新增測試：純 2D 噪聲 =====
    std::cout << "\n--- Generating Pure 2D Noise Patterns ---" << std::endl;
    generate2DOctaveBandNoise(IMAGE_SIZE, 5, "wavelet_noise_2D_octave_5.raw");
    generate2DOctaveBandNoise(IMAGE_SIZE, 6, "wavelet_noise_2D_octave_6.raw");
    generate2DOctaveBandNoise(IMAGE_SIZE, 7, "wavelet_noise_2D_octave_7.raw");

    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "Generated files for analysis:" << std::endl;
    std::cout << "  - wavelet_noise_3Dsliced_octave_*.raw: 3D noise sliced on a plane" << std::endl;
    std::cout << "  - wavelet_noise_2D_octave_*.raw: Pure 2D band-limited noise" << std::endl;
    std::cout << "\nUse the Python analysis script to visualize the band-limited property!" << std::endl;
    
    return 0;
}