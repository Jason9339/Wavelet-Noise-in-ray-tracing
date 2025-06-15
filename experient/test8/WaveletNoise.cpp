#include "WaveletNoise.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <limits>

// Static member initialization from Appendix 1
const float WaveletNoise::A_COEFFS[2 * WaveletNoise::ARAD] = {
    0.000334f, -0.001528f, 0.000410f, 0.003545f, -0.000938f, -0.008233f, 0.002172f, 0.019120f,
    -0.005040f, -0.044412f, 0.011655f, 0.103311f, -0.025936f, -0.243780f, 0.033979f, 0.655340f,
    0.655340f, 0.033979f, -0.243780f, -0.025936f, 0.103311f, 0.011655f, -0.044412f, -0.005040f,
    0.019120f, 0.002172f, -0.008233f, -0.000938f, 0.003546f, 0.000410f, -0.001528f, 0.000334f
};

const float WaveletNoise::P_COEFFS[4] = {0.25f, 0.75f, 0.75f, 0.25f};

WaveletNoise::WaveletNoise(int tileSize, unsigned int seed)
    : tileSizeN(tileSize), randomSeed(seed), rng(seed), gaussianDist(0.0f, 1.0f) {
    if (tileSizeN % 2 != 0) {
        tileSizeN++; // Paper, Appendix 1: "tile size must be even"
        std::cerr << "Warning: Tile size adjusted to " << tileSizeN << " (must be even)" << std::endl;
    }
}

WaveletNoise::~WaveletNoise() {}

// Modulo for periodic boundaries, from Appendix 1
int WaveletNoise::Mod(int x, int n) const {
    int m = x % n;
    return (m < 0) ? m + n : m;
}

// 1D Downsampling filter, from Appendix 1
void WaveletNoise::downsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride) {
    const float* a_center = &A_COEFFS[ARAD];
    
    for (int i = 0; i < n / 2; ++i) {
        float sum = 0.0f;
        for (int k = -ARAD; k < ARAD; ++k) {
            int from_idx = Mod(2 * i + k, n);
            sum += a_center[k] * from[from_idx * stride];
        }
        to[i * stride] = sum;
    }
}

// 1D Upsampling filter, from Appendix 1
void WaveletNoise::upsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride) {
    const float* p_center = &P_COEFFS[2];
    int n_coarse = n / 2;
    
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int k = i/2; k <= i/2 + 1; ++k) {
            int p_idx = i - 2 * k;
            if (p_idx >= -2 && p_idx <= 1) {
                int from_idx = Mod(k, n_coarse);
                sum += p_center[p_idx] * from[from_idx * stride];
            }
        }
        to[i * stride] = sum;
    }
}

// Generate 2D noise tile, following Section 3.6
void WaveletNoise::generateNoiseTile2D() {
    int n = tileSizeN;
    int num_elements = n * n;

    // Step 1: Fill the tile with random numbers (Gaussian as per Section 4.2)
    std::vector<float> r_initial(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        r_initial[i] = gaussianDist(rng);
    }
    
    std::vector<float> temp1(num_elements);
    std::vector<float> temp2(num_elements);
    
    // Create 1D buffers for row/column processing
    std::vector<float> line_in(n), line_ds(n / 2), line_out(n);
    
    // Steps 2 & 3: Separable filtering (Downsample and Upsample)
    // Process along X axis (rows)
    for (int iy = 0; iy < n; ++iy) {
        for(int ix = 0; ix < n; ++ix) line_in[ix] = r_initial[ix + iy * n];
        downsample1D(line_in, line_ds, n, 1);
        upsample1D(line_ds, line_out, n, 1);
        for(int ix = 0; ix < n; ++ix) temp1[ix + iy * n] = line_out[ix];
    }
    
    // Process along Y axis (columns) on the result of X-axis filtering
    for (int ix = 0; ix < n; ++ix) {
        for(int iy = 0; iy < n; ++iy) line_in[iy] = temp1[ix + iy * n];
        downsample1D(line_in, line_ds, n, 1);
        upsample1D(line_ds, line_out, n, 1);
        for(int iy = 0; iy < n; ++iy) temp2[ix + iy * n] = line_out[iy];
    }
    // temp2 now contains the low-pass version R_downsample_upsample

    // Step 4: Subtract to get N = R - R_downsample_upsample
    noiseCoefficients.resize(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] = r_initial[i] - temp2[i];
    }
}

// Evaluate 2D noise using 3x3 quadratic B-spline interpolation
float WaveletNoise::evaluate2D(const float p[2]) const {
    if (noiseCoefficients.empty()) return 0.0f;
    int n = static_cast<int>(std::round(std::sqrt(noiseCoefficients.size())));
    if (n == 0) return 0.0f;

    float result = 0.0f;
    int f[2], c[2], mid[2];
    float w[2][3];
    
    // Evaluate quadratic B-spline basis functions (from Appendix 2)
    for (int i = 0; i < 2; ++i) {
        mid[i] = static_cast<int>(std::ceil(p[i] - 0.5f));
        float t = mid[i] - (p[i] - 0.5f);
        w[i][0] = t * t / 2.0f;
        w[i][2] = (1.0f - t) * (1.0f - t) / 2.0f;
        w[i][1] = 1.0f - w[i][0] - w[i][2];
    }
    
    // Sum contributions from 3x3=9 neighboring coefficients
    for (f[1] = -1; f[1] <= 1; ++f[1]) {
        for (f[0] = -1; f[0] <= 1; ++f[0]) {
            float weight = w[0][f[0] + 1] * w[1][f[1] + 1];
            c[0] = Mod(mid[0] + f[0], n);
            c[1] = Mod(mid[1] + f[1], n);
            int idx = c[0] + c[1] * n;
            result += weight * noiseCoefficients[idx];
        }
    }
    return result;
}

void WaveletNoise::generateNoiseTile3D() {
    int n = tileSizeN;
    int num_elements = n * n * n;
    
    std::vector<float> r_initial(num_elements);
    for (int i = 0; i < num_elements; ++i) r_initial[i] = gaussianDist(rng);
    
    std::vector<float> temp1(num_elements), temp2(num_elements);
    std::vector<float> line_in(n), line_ds(n / 2), line_out(n);
    
    // Process along X axis
    for (int iz = 0; iz < n; ++iz) for (int iy = 0; iy < n; ++iy) {
        int base = iy * n + iz * n * n;
        for(int ix = 0; ix < n; ++ix) line_in[ix] = r_initial[base + ix];
        downsample1D(line_in, line_ds, n, 1);
        upsample1D(line_ds, line_out, n, 1);
        for(int ix = 0; ix < n; ++ix) temp1[base + ix] = line_out[ix];
    }
    
    // Process along Y axis
    for (int iz = 0; iz < n; ++iz) for (int ix = 0; ix < n; ++ix) {
        int base = ix + iz * n * n;
        for(int iy = 0; iy < n; ++iy) line_in[iy] = temp1[base + iy * n];
        downsample1D(line_in, line_ds, n, 1);
        upsample1D(line_ds, line_out, n, 1);
        for(int iy = 0; iy < n; ++iy) temp2[base + iy * n] = line_out[iy];
    }
    
    // Process along Z axis
    for (int iy = 0; iy < n; ++iy) for (int ix = 0; ix < n; ++ix) {
        int base = ix + iy * n;
        for(int iz = 0; iz < n; ++iz) line_in[iz] = temp2[base + iz * n * n];
        downsample1D(line_in, line_ds, n, 1);
        upsample1D(line_ds, line_out, n, 1);
        for(int iz = 0; iz < n; ++iz) temp1[base + iz * n * n] = line_out[iz];
    }

    noiseCoefficients.resize(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] = r_initial[i] - temp1[i];
    }
}

float WaveletNoise::evaluate3D(const float p[3]) const {
    if (noiseCoefficients.empty()) return 0.0f;
    int n = static_cast<int>(std::round(std::cbrt(noiseCoefficients.size())));
    if (n == 0) return 0.0f;

    float result = 0.0f;
    int f[3], c[3], mid[3];
    float w[3][3];
    
    for (int i = 0; i < 3; ++i) {
        mid[i] = static_cast<int>(std::ceil(p[i] - 0.5f));
        float t = mid[i] - (p[i] - 0.5f);
        w[i][0] = t * t / 2.0f;
        w[i][2] = (1.0f - t) * (1.0f - t) / 2.0f;
        w[i][1] = 1.0f - w[i][0] - w[i][2];
    }
    
    for (f[2] = -1; f[2] <= 1; ++f[2]) {
        for (f[1] = -1; f[1] <= 1; ++f[1]) {
            for (f[0] = -1; f[0] <= 1; ++f[0]) {
                float weight = w[0][f[0] + 1] * w[1][f[1] + 1] * w[2][f[2] + 1];
                c[0] = Mod(mid[0] + f[0], n);
                c[1] = Mod(mid[1] + f[1], n);
                c[2] = Mod(mid[2] + f[2], n);
                int idx = c[0] + c[1] * n + c[2] * n * n;
                result += weight * noiseCoefficients[idx];
            }
        }
    }
    return result;
}

// Section 3.7 and Appendix 2: Projecting 3D noise onto a 2D surface
float WaveletNoise::evaluate3DProjected(const float p[3], const float normal[3]) const {
    if (noiseCoefficients.empty()) return 0.0f;
    int n = static_cast<int>(std::round(std::cbrt(noiseCoefficients.size())));
    if (n == 0) return 0.0f;
    
    float result = 0.0f;
    int c[3];
    
    // Appendix 2: Bounding the support of the basis functions
    int min_c[3], max_c[3];
    for (int i = 0; i < 3; ++i) {
        float support = 3.0f * std::abs(normal[i]) + 3.0f * std::sqrt((1.0f - normal[i] * normal[i]) / 2.0f);
        min_c[i] = static_cast<int>(std::ceil(p[i] - support));
        max_c[i] = static_cast<int>(std::floor(p[i] + support));
    }
    
    // Loop over the noise coefficients within the bound
    for (c[2] = min_c[2]; c[2] <= max_c[2]; ++c[2]) {
        for (c[1] = min_c[1]; c[1] <= max_c[1]; ++c[1]) {
            for (c[0] = min_c[0]; c[0] <= max_c[0]; ++c[0]) {
                // Dot the normal with the vector from coefficient c to point p
                float dot = 0.0f;
                for (int i = 0; i < 3; ++i) dot += normal[i] * (p[i] - c[i]);
                
                // Evaluate the basis function at c moved halfway to p along the normal
                float weight = 1.0f;
                for (int i = 0; i < 3; ++i) {
                    float t = (static_cast<float>(c[i]) + normal[i] * dot / 2.0f) - (p[i] - 1.5f);
                    
                    if (t <= 0.0f || t >= 3.0f) {
                        weight = 0.0f;
                        break;
                    }
                    float t1 = t - 1.0f, t2 = 2.0f - t, t3 = 3.0f - t;
                    if (t < 1.0f)       weight *= (t * t / 2.0f);
                    else if (t < 2.0f)  weight *= (1.0f - (t1 * t1 + t2 * t2) / 2.0f);
                    else                weight *= (t3 * t3 / 2.0f);
                }
                
                if (weight > 1e-6) {
                    int idx = Mod(c[0], n) + Mod(c[1], n) * n + Mod(c[2], n) * n * n;
                    result += weight * noiseCoefficients[idx];
                }
            }
        }
    }
    return result;
}

// Statistical analysis functions for variance verification (Section 4.2)
DataStats WaveletNoise::calculateStats(const std::vector<float>& data, const std::string& name) const {
    DataStats stats;
    if (data.empty()) return stats;
    
    double sum = 0.0, sum_sq = 0.0;
    for (const float& val : data) {
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
    }
    
    stats.avg = static_cast<float>(sum / data.size());
    stats.var = static_cast<float>((sum_sq / data.size()) - static_cast<double>(stats.avg) * stats.avg);
    
    std::cout << name << " stats: "
              << "avg=" << stats.avg << ", "
              << "var=" << stats.var << ", "
              << "stddev=" << std::sqrt(stats.var) << std::endl;
    return stats;
}

const std::vector<float>& WaveletNoise::getNoiseCoefficients() const { return noiseCoefficients; }
int WaveletNoise::getTileSize() const { return tileSizeN; }