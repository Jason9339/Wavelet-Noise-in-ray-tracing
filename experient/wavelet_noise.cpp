// wavelet_noise.cpp
// Complete implementation of "Wavelet Noise" (Cook & DeRose, 2005)
// 完整實現論文中的所有算法，包括3D噪音、投影等

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <complex>
#include <cstring>

// ===== BMP 檔案輸出工具 =====
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t type = 0x4D42;  // "BM"
    uint32_t size;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset = 54;
    uint32_t dib_header_size = 40;
    int32_t width;
    int32_t height;
    uint16_t planes = 1;
    uint16_t bits_per_pixel = 24;
    uint32_t compression = 0;
    uint32_t image_size = 0;
    int32_t x_pixels_per_meter = 0;
    int32_t y_pixels_per_meter = 0;
    uint32_t colors_used = 0;
    uint32_t important_colors = 0;
};
#pragma pack(pop)

void saveBMP(const std::string& filename, const std::vector<std::vector<float>>& data) {
    int width = data[0].size();
    int height = data.size();
    
    BMPHeader header;
    header.width = width;
    header.height = height;
    header.size = 54 + 3 * width * height;
    
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<char*>(&header), sizeof(header));
    
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            uint8_t value = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (data[y][x] + 1.0f) * 127.5f)));
            file.write(reinterpret_cast<char*>(&value), 1); // B
            file.write(reinterpret_cast<char*>(&value), 1); // G
            file.write(reinterpret_cast<char*>(&value), 1); // R
        }
        int padding = (4 - (width * 3) % 4) % 4;
        for (int i = 0; i < padding; i++) {
            uint8_t pad = 0;
            file.write(reinterpret_cast<char*>(&pad), 1);
        }
    }
    
    file.close();
    std::cout << "Saved: " << filename << std::endl;
}

// ===== 數學工具函數 =====
inline int Mod(int x, int n) {
    int m = x % n;
    return (m < 0) ? m + n : m;
}

// Box-Muller 變換生成高斯隨機數
float gaussianNoise() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

// ===== 完整的 Wavelet Noise 實現（基於論文附錄） =====
class WaveletNoise {
private:
    static float* noiseTileData;
    static int noiseTileSize;
    
    // 下採樣係數（論文附錄1）
    static constexpr int ARAD = 16;
    
public:
    static void GenerateNoiseTile(int n, int olap = 0) {
        if (n % 2) n++; // tile size must be even
        
        int sz = n * n * n * sizeof(float);
        float* temp1 = (float*)malloc(sz);
        float* temp2 = (float*)malloc(sz);
        float* noise = (float*)malloc(sz);
        
        // Step 1. Fill the tile with random numbers in the range -1 to 1
        for (int i = 0; i < n * n * n; i++) {
            noise[i] = gaussianNoise();
        }
        
        // Steps 2 and 3. Downsample and upsample the tile
        // Each x row
        for (int iy = 0; iy < n; iy++) {
            for (int iz = 0; iz < n; iz++) {
                int i = iy * n + iz * n * n;
                Downsample(&noise[i], &temp1[i], n, 1);
                Upsample(&temp1[i], &temp2[i], n, 1);
            }
        }
        
        // Each y row
        for (int ix = 0; ix < n; ix++) {
            for (int iz = 0; iz < n; iz++) {
                int i = ix + iz * n * n;
                Downsample(&temp2[i], &temp1[i], n, n);
                Upsample(&temp1[i], &temp2[i], n, n);
            }
        }
        
        // Each z row
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                int i = ix + iy * n;
                Downsample(&temp2[i], &temp1[i], n, n * n);
                Upsample(&temp1[i], &temp2[i], n, n * n);
            }
        }
        
        // Step 4. Subtract out the coarse-scale contribution
        for (int i = 0; i < n * n * n; i++) {
            noise[i] -= temp2[i];
        }
        
        // Avoid even/odd variance difference by adding odd-offset version
        int offset = n / 2;
        if (offset % 2 == 0) offset++;
        
        int i = 0;
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                for (int iz = 0; iz < n; iz++) {
                    temp1[i++] = noise[Mod(ix + offset, n) + 
                                      Mod(iy + offset, n) * n + 
                                      Mod(iz + offset, n) * n * n];
                }
            }
        }
        
        for (int i = 0; i < n * n * n; i++) {
            noise[i] += temp1[i];
        }
        
        noiseTileData = noise;
        noiseTileSize = n;
        
        free(temp1);
        free(temp2);
    }
    
    static void Downsample(float* from, float* to, int n, int stride) {
        float aCoeffs[2*ARAD] = {
            0.000334,-0.001528, 0.000410, 0.003545,-0.000938,-0.008233, 0.002172, 0.019120,
            -0.005040,-0.044412, 0.011655, 0.103311,-0.025936,-0.243780, 0.033979, 0.655340,
            0.655340, 0.033979,-0.243780,-0.025936, 0.103311, 0.011655,-0.044412,-0.005040,
            0.019120, 0.002172,-0.008233,-0.000938, 0.003546, 0.000410,-0.001528, 0.000334
        };
        
        float* a = &aCoeffs[ARAD];
        
        for (int i = 0; i < n/2; i++) {
            to[i*stride] = 0;
            for (int k = 2*i - ARAD; k <= 2*i + ARAD; k++) {
                to[i*stride] += a[k - 2*i] * from[Mod(k, n) * stride];
            }
        }
    }
    
    static void Upsample(float* from, float* to, int n, int stride) {
        float pCoeffs[4] = { 0.25, 0.75, 0.75, 0.25 };
        float* p = &pCoeffs[2];
        
        for (int i = 0; i < n; i++) {
            to[i*stride] = 0;
            for (int k = i/2; k <= i/2 + 1; k++) {
                to[i*stride] += p[i - 2*k] * from[Mod(k, n/2) * stride];
            }
        }
    }
    
    // Non-projected 3D noise (論文附錄2)
    static float WNoise(float p[3]) {
        int i, f[3], c[3], mid[3], n = noiseTileSize;
        float w[3][3], t, result = 0;
        
        // Evaluate quadratic B-spline basis functions
        for (i = 0; i < 3; i++) {
            mid[i] = ceil(p[i] - 0.5);
            t = mid[i] - (p[i] - 0.5);
            w[i][0] = t * t / 2;
            w[i][2] = (1 - t) * (1 - t) / 2;
            w[i][1] = 1 - w[i][0] - w[i][2];
        }
        
        // Evaluate noise by weighting noise coefficients by basis function values
        for (f[2] = -1; f[2] <= 1; f[2]++) {
            for (f[1] = -1; f[1] <= 1; f[1]++) {
                for (f[0] = -1; f[0] <= 1; f[0]++) {
                    float weight = 1;
                    for (i = 0; i < 3; i++) {
                        c[i] = Mod(mid[i] + f[i], n);
                        weight *= w[i][f[i] + 1];
                    }
                    result += weight * noiseTileData[c[2] * n * n + c[1] * n + c[0]];
                }
            }
        }
        
        return result;
    }
    
    // 3D noise projected onto 2D (論文附錄2)
    static float WProjectedNoise(float p[3], float normal[3]) {
        int i, c[3], min[3], max[3], n = noiseTileSize;
        float support, result = 0;
        
        // Bound the support of the basis functions for this projection direction
        for (i = 0; i < 3; i++) {
            support = 3 * fabs(normal[i]) + 3 * sqrt((1 - normal[i] * normal[i]) / 2);
            min[i] = ceil(p[i] - support);
            max[i] = floor(p[i] + support);
        }
        
        // Loop over the noise coefficients within the bound
        for (c[2] = min[2]; c[2] <= max[2]; c[2]++) {
            for (c[1] = min[1]; c[1] <= max[1]; c[1]++) {
                for (c[0] = min[0]; c[0] <= max[0]; c[0]++) {
                    float t, t1, t2, t3, dot = 0, weight = 1;
                    
                    // Dot the normal with the vector from c to p
                    for (i = 0; i < 3; i++) {
                        dot += normal[i] * (p[i] - c[i]);
                    }
                    
                    // Evaluate the basis function at c moved halfway to p along the normal
                    for (i = 0; i < 3; i++) {
                        t = (c[i] + normal[i] * dot / 2) - (p[i] - 1.5);
                        t1 = t - 1;
                        t2 = 2 - t;
                        t3 = 3 - t;
                        weight *= (t <= 0 || t >= 3) ? 0 : 
                                 (t < 1) ? t * t / 2 : 
                                 (t < 2) ? 1 - (t1 * t1 + t2 * t2) / 2 : 
                                 t3 * t3 / 2;
                    }
                    
                    // Evaluate noise by weighting noise coefficients by basis function values
                    result += weight * noiseTileData[Mod(c[2], n) * n * n + 
                                                   Mod(c[1], n) * n + 
                                                   Mod(c[0], n)];
                }
            }
        }
        
        return result;
    }
    
    // Multiband noise (論文附錄2)
    static float WMultibandNoise(float p[3], float s, float* normal, 
                                int firstBand, int nbands, float* w) {
        float q[3], result = 0, variance = 0;
        int i, b;
        
        for (b = 0; b < nbands && s + firstBand + b < 0; b++) {
            for (i = 0; i <= 2; i++) {
                q[i] = 2 * p[i] * pow(2, firstBand + b);
            }
            result += (normal) ? w[b] * WProjectedNoise(q, normal) : w[b] * WNoise(q);
        }
        
        for (b = 0; b < nbands; b++) {
            variance += w[b] * w[b];
        }
        
        // Adjust the noise so it has a variance of 1
        if (variance) {
            result /= sqrt(variance * ((normal) ? 0.296 : 0.210));
        }
        
        return result;
    }
    
    // 2D convenience functions
    static float Noise2D(float x, float y) {
        float p[3] = {x, y, 0.5f};
        return WNoise(p);
    }
    
    static float MultibandNoise2D(float x, float y, int nbands, float* weights) {
        float p[3] = {x, y, 0.5f};
        return WMultibandNoise(p, 0, nullptr, -nbands+1, nbands, weights);
    }
};

// Static member definitions
float* WaveletNoise::noiseTileData = nullptr;
int WaveletNoise::noiseTileSize = 0;

// ===== Perlin Noise 實現（用於比較） =====
class PerlinNoise {
private:
    std::vector<int> p;
    
    float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    float lerp(float t, float a, float b) {
        return a + t * (b - a);
    }
    
    float grad(int hash, float x, float y, float z) {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    
public:
    PerlinNoise() {
        p.resize(512);
        std::iota(p.begin(), p.begin() + 256, 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(p.begin(), p.begin() + 256, g);
        std::copy(p.begin(), p.begin() + 256, p.begin() + 256);
    }
    
    float noise(float x, float y, float z) {
        int X = (int)floor(x) & 255;
        int Y = (int)floor(y) & 255;
        int Z = (int)floor(z) & 255;
        
        x -= floor(x);
        y -= floor(y);
        z -= floor(z);
        
        float u = fade(x);
        float v = fade(y);
        float w = fade(z);
        
        int A = p[X] + Y;
        int AA = p[A] + Z;
        int AB = p[A + 1] + Z;
        int B = p[X + 1] + Y;
        int BA = p[B] + Z;
        int BB = p[B + 1] + Z;
        
        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                       grad(p[BA], x - 1, y, z)),
                              lerp(u, grad(p[AB], x, y - 1, z),
                                       grad(p[BB], x - 1, y - 1, z))),
                      lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
                                       grad(p[BA + 1], x - 1, y, z - 1)),
                              lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                                       grad(p[BB + 1], x - 1, y - 1, z - 1))));
    }
    
    float noise2D(float x, float y) {
        return noise(x, y, 0.5f);
    }
};

// ===== FFT 實現 =====
using Complex = std::complex<float>;

void fft2D(const std::vector<std::vector<float>>& input,
           std::vector<std::vector<Complex>>& output) {
    int height = input.size();
    int width = input[0].size();
    
    output.resize(height, std::vector<Complex>(width));
    
    // DFT implementation
    for (int k = 0; k < height; k++) {
        for (int l = 0; l < width; l++) {
            Complex sum(0, 0);
            
            for (int m = 0; m < height; m++) {
                for (int n = 0; n < width; n++) {
                    float angle = -2 * M_PI * ((float)(k * m) / height + (float)(l * n) / width);
                    sum += input[m][n] * Complex(cos(angle), sin(angle));
                }
            }
            
            output[k][l] = sum;
        }
    }
}

void saveSpectrum(const std::string& filename, const std::vector<std::vector<Complex>>& spectrum) {
    int height = spectrum.size();
    int width = spectrum[0].size();
    std::vector<std::vector<float>> magnitude(height, std::vector<float>(width));
    
    float maxMag = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            magnitude[i][j] = std::abs(spectrum[i][j]);
            maxMag = std::max(maxMag, magnitude[i][j]);
        }
    }
    
    // FFT shift and log scale
    std::vector<std::vector<float>> output(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int shiftedI = (i + height/2) % height;
            int shiftedJ = (j + width/2) % width;
            
            float mag = magnitude[shiftedI][shiftedJ];
            float logMag = log(1 + mag / maxMag * 255);
            output[i][j] = logMag / log(256) * 2 - 1;
        }
    }
    
    saveBMP(filename, output);
}

// ===== 測試函數 =====
void generateNoiseComparison(int imageSize = 256) {
    std::cout << "Generating noise patterns for comparison..." << std::endl;
    
    // 2D noise patterns
    std::vector<std::vector<float>> wavelet2D(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> perlin2D(imageSize, std::vector<float>(imageSize));
    
    PerlinNoise perlin;
    
    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            float fx = x / 32.0f;
            float fy = y / 32.0f;
            
            wavelet2D[y][x] = WaveletNoise::Noise2D(fx, fy);
            perlin2D[y][x] = perlin.noise2D(fx, fy);
        }
    }
    
    saveBMP("wavelet_2d.bmp", wavelet2D);
    saveBMP("perlin_2d.bmp", perlin2D);
    
    // 3D slice comparison
    std::vector<std::vector<float>> wavelet3DSlice(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> perlin3DSlice(imageSize, std::vector<float>(imageSize));
    
    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            float p[3] = {x / 32.0f, y / 32.0f, 0.5f};
            wavelet3DSlice[y][x] = WaveletNoise::WNoise(p);
            perlin3DSlice[y][x] = perlin.noise(p[0], p[1], p[2]);
        }
    }
    
    saveBMP("wavelet_3d_slice.bmp", wavelet3DSlice);
    saveBMP("perlin_3d_slice.bmp", perlin3DSlice);
    
    // 3D projection comparison
    std::vector<std::vector<float>> waveletProjection(imageSize, std::vector<float>(imageSize));
    float normal[3] = {0, 0, 1}; // Project along Z axis
    
    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            float p[3] = {x / 32.0f, y / 32.0f, 0.0f};
            waveletProjection[y][x] = WaveletNoise::WProjectedNoise(p, normal);
        }
    }
    
    saveBMP("wavelet_3d_projected.bmp", waveletProjection);
    
    // Multiband noise
    float weights4[4] = {0.5f, 0.25f, 0.125f, 0.0625f};
    std::vector<std::vector<float>> waveletMulti(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> perlinMulti(imageSize, std::vector<float>(imageSize));
    
    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            waveletMulti[y][x] = WaveletNoise::MultibandNoise2D(x / 32.0f, y / 32.0f, 4, weights4);
            
            // Perlin multiband
            float sum = 0, variance = 0;
            for (int b = 0; b < 4; b++) {
                float scale = pow(2.0f, b);
                sum += weights4[b] * perlin.noise2D(x / 32.0f * scale, y / 32.0f * scale);
                variance += weights4[b] * weights4[b];
            }
            perlinMulti[y][x] = sum / sqrt(variance);
        }
    }
    
    saveBMP("wavelet_multiband.bmp", waveletMulti);
    saveBMP("perlin_multiband.bmp", perlinMulti);
}

void generateSpectralAnalysis(int imageSize = 256) {
    std::cout << "Performing spectral analysis..." << std::endl;
    
    // Generate single band noise for spectral analysis
    std::vector<std::vector<float>> wavelet(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> perlin(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> waveletSlice(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> waveletProjected(imageSize, std::vector<float>(imageSize));
    
    PerlinNoise perlinGen;
    float normal[3] = {0, 0, 1};
    
    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            float fx = x / 32.0f;
            float fy = y / 32.0f;
            
            // 2D noise
            wavelet[y][x] = WaveletNoise::Noise2D(fx, fy);
            perlin[y][x] = perlinGen.noise2D(fx, fy);
            
            // 3D slice
            float p[3] = {fx, fy, 0.5f};
            waveletSlice[y][x] = WaveletNoise::WNoise(p);
            
            // 3D projected
            p[2] = 0.0f;
            waveletProjected[y][x] = WaveletNoise::WProjectedNoise(p, normal);
        }
    }
    
    // Compute spectra
    std::vector<std::vector<Complex>> waveletSpectrum, perlinSpectrum, sliceSpectrum, projectedSpectrum;
    
    fft2D(wavelet, waveletSpectrum);
    fft2D(perlin, perlinSpectrum);
    fft2D(waveletSlice, sliceSpectrum);
    fft2D(waveletProjected, projectedSpectrum);
    
    saveSpectrum("spectrum_wavelet_2d.bmp", waveletSpectrum);
    saveSpectrum("spectrum_perlin_2d.bmp", perlinSpectrum);
    saveSpectrum("spectrum_wavelet_3d_slice.bmp", sliceSpectrum);
    saveSpectrum("spectrum_wavelet_3d_projected.bmp", projectedSpectrum);
}

void generateBandComparison() {
    std::cout << "Generating individual frequency bands..." << std::endl;
    
    int imageSize = 256;
    float singleWeight[1] = {1.0f};
    
    // Generate 3 adjacent bands for both Wavelet and Perlin
    for (int band = -2; band <= 0; band++) {
        std::vector<std::vector<float>> waveletBand(imageSize, std::vector<float>(imageSize));
        std::vector<std::vector<float>> perlinBand(imageSize, std::vector<float>(imageSize));
        
        PerlinNoise perlin;
        float scale = pow(2.0f, -band);
        
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                float p[3] = {x / 32.0f * scale, y / 32.0f * scale, 0.5f};
                waveletBand[y][x] = WaveletNoise::WMultibandNoise(p, 0, nullptr, band, 1, singleWeight);
                perlinBand[y][x] = perlin.noise(p[0], p[1], p[2]);
            }
        }
        
        saveBMP("wavelet_band_" + std::to_string(-band) + ".bmp", waveletBand);
        saveBMP("perlin_band_" + std::to_string(-band) + ".bmp", perlinBand);
        
        // Compute spectrum for each band
        std::vector<std::vector<Complex>> waveletSpec, perlinSpec;
        fft2D(waveletBand, waveletSpec);
        fft2D(perlinBand, perlinSpec);
        
        saveSpectrum("spectrum_wavelet_band_" + std::to_string(-band) + ".bmp", waveletSpec);
        saveSpectrum("spectrum_perlin_band_" + std::to_string(-band) + ".bmp", perlinSpec);
    }
}

// ===== 主程式 =====
int main() {
    std::cout << "=== Wavelet Noise Complete Implementation ===" << std::endl;
    std::cout << "Based on 'Wavelet Noise' by Cook & DeRose (2005)" << std::endl << std::endl;
    
    // Initialize noise tile
    std::cout << "Generating noise tile..." << std::endl;
    WaveletNoise::GenerateNoiseTile(32);
    
    // Generate all comparisons
    generateNoiseComparison();
    generateSpectralAnalysis();
    generateBandComparison();
    
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "\nVisual Comparison:" << std::endl;
    std::cout << "- wavelet_2d.bmp / perlin_2d.bmp: 2D noise comparison" << std::endl;
    std::cout << "- wavelet_3d_slice.bmp / perlin_3d_slice.bmp: 3D noise slice" << std::endl;
    std::cout << "- wavelet_3d_projected.bmp: 3D noise projected to 2D" << std::endl;
    std::cout << "- wavelet_multiband.bmp / perlin_multiband.bmp: Multiband noise" << std::endl;
    
    std::cout << "\nSpectral Analysis:" << std::endl;
    std::cout << "- spectrum_wavelet_2d.bmp / spectrum_perlin_2d.bmp: 2D noise spectra" << std::endl;
    std::cout << "- spectrum_wavelet_3d_slice.bmp: 3D slice spectrum" << std::endl;
    std::cout << "- spectrum_wavelet_3d_projected.bmp: 3D projected spectrum" << std::endl;
    
    std::cout << "\nFrequency Band Analysis:" << std::endl;
    std::cout << "- wavelet_band_0,1,2.bmp / perlin_band_0,1,2.bmp: Individual bands" << std::endl;
    std::cout << "- spectrum_wavelet_band_0,1,2.bmp / spectrum_perlin_band_0,1,2.bmp: Band spectra" << std::endl;
    
    return 0;
}