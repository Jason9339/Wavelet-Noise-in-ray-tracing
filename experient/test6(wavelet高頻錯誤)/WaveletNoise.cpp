#include "WaveletNoise.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <limits>

// Static member initialization
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
        tileSizeN++; // Paper: "tile size must be even"
        std::cerr << "Warning: Tile size adjusted to " << tileSizeN << " (must be even)" << std::endl;
    }
}

WaveletNoise::~WaveletNoise() {}

int WaveletNoise::Mod(int x, int n) const {
    int m = x % n;
    return (m < 0) ? m + n : m;
}

void WaveletNoise::downsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride) {
    const float* a_center = &A_COEFFS[ARAD];
    
    for (int i = 0; i < n / 2; ++i) {
        float sum = 0.0f;
        for (int k = -ARAD; k < ARAD; ++k) {
            int idx = 2 * i + k;
            sum += a_center[k] * from[Mod(idx, n) * stride];
        }
        to[i * stride] = sum;
    }
}

void WaveletNoise::upsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride) {
    const float* p_center = &P_COEFFS[2];
    int n_coarse = n / 2;
    
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int k = i/2; k <= i/2 + 1; ++k) {
            int p_idx = i - 2 * k;
            if (p_idx >= -2 && p_idx <= 1) {
                sum += p_center[p_idx] * from[Mod(k, n_coarse) * stride];
            }
        }
        to[i * stride] = sum;
    }
}

void WaveletNoise::generateNoiseTile3D() {
    int n = tileSizeN;
    if (n % 2) n++;
    
    int num_elements = n * n * n;
    noiseCoefficients.assign(num_elements, 0.0f);
    
    // Allocate working arrays
    std::vector<float> noise(num_elements);      // Initial random data
    std::vector<float> temp1(num_elements);      // Working array 1
    std::vector<float> temp2(num_elements);      // Working array 2
    
    // Step 1: Fill with random numbers
    for (int i = 0; i < num_elements; ++i) {
        noise[i] = gaussianDist(rng);
    }
    saveIntermediateData(noise, n, DebugStep::R_INITIAL);
    
    // Create temporary 1D arrays for processing
    std::vector<float> line_in(n);
    std::vector<float> line_ds(n / 2);
    std::vector<float> line_out(n);
    
    // Process along X axis: noise -> temp1
    std::copy(noise.begin(), noise.end(), temp1.begin());
    for (int iy = 0; iy < n; ++iy) {
        for (int iz = 0; iz < n; ++iz) {
            int base = iy * n + iz * n * n;
            
            // Extract line
            for (int ix = 0; ix < n; ++ix) {
                line_in[ix] = noise[base + ix];
            }
            
            // Downsample and upsample
            downsample1D(line_in, line_ds, n, 1);
            upsample1D(line_ds, line_out, n, 1);
            
            // Store result
            for (int ix = 0; ix < n; ++ix) {
                temp1[base + ix] = line_out[ix];
            }
        }
    }
    saveIntermediateData(temp1, n, DebugStep::AFTER_X_FILTER);
    
    // Process along Y axis: temp1 -> temp2
    std::copy(temp1.begin(), temp1.end(), temp2.begin());
    for (int ix = 0; ix < n; ++ix) {
        for (int iz = 0; iz < n; ++iz) {
            int base = ix + iz * n * n;
            
            // Extract line
            for (int iy = 0; iy < n; ++iy) {
                line_in[iy] = temp1[base + iy * n];
            }
            
            // Downsample and upsample
            downsample1D(line_in, line_ds, n, 1);
            upsample1D(line_ds, line_out, n, 1);
            
            // Store result
            for (int iy = 0; iy < n; ++iy) {
                temp2[base + iy * n] = line_out[iy];
            }
        }
    }
    saveIntermediateData(temp2, n, DebugStep::AFTER_Y_FILTER);
    
    // Process along Z axis: temp2 -> temp1
    std::copy(temp2.begin(), temp2.end(), temp1.begin());
    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            int base = ix + iy * n;
            
            // Extract line
            for (int iz = 0; iz < n; ++iz) {
                line_in[iz] = temp2[base + iz * n * n];
            }
            
            // Downsample and upsample
            downsample1D(line_in, line_ds, n, 1);
            upsample1D(line_ds, line_out, n, 1);
            
            // Store result
            for (int iz = 0; iz < n; ++iz) {
                temp1[base + iz * n * n] = line_out[iz];
            }
        }
    }
    saveIntermediateData(temp1, n, DebugStep::AFTER_Z_FILTER);
    
    // Step 4: Subtract to get N = R - R_downsample_upsample
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] = noise[i] - temp1[i];
    }
    saveIntermediateData(noiseCoefficients, n, DebugStep::N_COEFFS_PRE_CORRECTION);
    
    // Avoid even/odd variance difference
    int offset = n / 2;
    if (offset % 2 == 0) offset++;
    
    std::vector<float> shifted(num_elements);
    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            for (int iz = 0; iz < n; ++iz) {
                int idx = ix + iy * n + iz * n * n;
                int shifted_idx = Mod(ix + offset, n) +
                                 Mod(iy + offset, n) * n +
                                 Mod(iz + offset, n) * n * n;
                shifted[idx] = noiseCoefficients[shifted_idx];
            }
        }
    }
    
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] += shifted[i];
    }
    saveIntermediateData(noiseCoefficients, n, DebugStep::N_COEFFS_FINAL);
}

void WaveletNoise::generateNoiseTile() {
    generateNoiseTile3D();
}

float WaveletNoise::evaluate3D(const float p[3]) const {
    float result = 0.0f;
    int f[3], c[3], mid[3];
    float w[3][3];
    
    int n = static_cast<int>(std::round(std::cbrt(noiseCoefficients.size())));
    if (n == 0 || noiseCoefficients.empty()) return 0.0f;
    
    // Evaluate quadratic B-spline basis functions
    for (int i = 0; i < 3; ++i) {
        mid[i] = static_cast<int>(std::ceil(p[i] - 0.5f));
        float t = mid[i] - (p[i] - 0.5f);
        w[i][0] = t * t / 2.0f;
        w[i][2] = (1.0f - t) * (1.0f - t) / 2.0f;
        w[i][1] = 1.0f - w[i][0] - w[i][2];
    }
    
    // Sum contributions from 27 neighboring coefficients
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

float WaveletNoise::evaluate3DProjected(const float p[3], const float normal[3]) const {
    if (noiseCoefficients.empty()) return 0.0f;
    
    int n = static_cast<int>(std::round(std::cbrt(noiseCoefficients.size())));
    if (n == 0) return 0.0f;
    
    float result = 0.0f;
    int c[3];
    
    // Bound the support
    int min_c[3], max_c[3];
    for (int i = 0; i < 3; ++i) {
        float support = 3.0f * std::abs(normal[i]) + 3.0f * std::sqrt((1.0f - normal[i] * normal[i]) / 2.0f);
        min_c[i] = static_cast<int>(std::ceil(p[i] - support));
        max_c[i] = static_cast<int>(std::floor(p[i] + support));
    }
    
    // Loop over coefficients
    for (c[2] = min_c[2]; c[2] <= max_c[2]; ++c[2]) {
        for (c[1] = min_c[1]; c[1] <= max_c[1]; ++c[1]) {
            for (c[0] = min_c[0]; c[0] <= max_c[0]; ++c[0]) {
                float dot = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    dot += normal[i] * (p[i] - c[i]);
                }
                
                float weight = 1.0f;
                for (int i = 0; i < 3; ++i) {
                    float t = (static_cast<float>(c[i]) + normal[i] * dot / 2.0f) - (p[i] - 1.5f);
                    
                    if (t <= 0.0f || t >= 3.0f) {
                        weight = 0.0f;
                        break;
                    } else if (t < 1.0f) {
                        weight *= (t * t / 2.0f);
                    } else if (t < 2.0f) {
                        float t1 = t - 1.0f;
                        float t2 = 2.0f - t;
                        weight *= (1.0f - (t1 * t1 + t2 * t2) / 2.0f);
                    } else {
                        float t3 = 3.0f - t;
                        weight *= (t3 * t3 / 2.0f);
                    }
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

// Statistical analysis functions
DataStats WaveletNoise::calculateStats(const std::vector<float>& data, const std::string& name) const {
    DataStats stats;
    if (data.empty()) {
        std::cerr << "Warning: Empty data for stats: " << name << std::endl;
        return stats;
    }
    
    double sum = 0.0, sum_sq = 0.0;
    long long valid_count = 0;
    
    for (const float& val : data) {
        if (std::isnan(val) || std::isinf(val)) {
            stats.count_nan_inf++;
            continue;
        }
        valid_count++;
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
    }
    
    if (valid_count > 0) {
        stats.avg = static_cast<float>(sum / valid_count);
        stats.var = static_cast<float>((sum_sq / valid_count) - static_cast<double>(stats.avg) * stats.avg);
        stats.energy = static_cast<float>(sum_sq);
    }
    
    std::cout << name << " stats:"
              << " avg=" << stats.avg
              << " var=" << stats.var
              << " stddev=" << (stats.var > 0 ? std::sqrt(stats.var) : 0)
              << " min=" << stats.min_val
              << " max=" << stats.max_val
              << " energy=" << stats.energy;
    if (stats.count_nan_inf > 0) {
        std::cout << " (NaN/Inf: " << stats.count_nan_inf << ")";
    }
    std::cout << std::endl;
    
    return stats;
}

float WaveletNoise::calculateTotalEnergy(const std::vector<float>& data) const {
    double sum_sq = 0.0;
    for (float val : data) {
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
    }
    return static_cast<float>(sum_sq);
}

void WaveletNoise::saveIntermediateData(const std::vector<float>& data, int dim_n, DebugStep step, const std::string& base_filename) const {
    if (data.empty()) {
        std::cerr << "Data is empty for step " << static_cast<int>(step) << std::endl;
        return;
    }
    
    std::ostringstream oss;
    oss << base_filename;
    switch (step) {
        case DebugStep::R_INITIAL: oss << "0_R_initial"; break;
        case DebugStep::AFTER_X_FILTER: oss << "1_R_ds_us_x"; break;
        case DebugStep::AFTER_Y_FILTER: oss << "2_R_ds_us_xy"; break;
        case DebugStep::AFTER_Z_FILTER: oss << "3_R_ds_us_xyz"; break;
        case DebugStep::N_COEFFS_PRE_CORRECTION: oss << "4_N_coeffs_pre_correct"; break;
        case DebugStep::N_COEFFS_FINAL: oss << "5_N_coeffs_final"; break;
        default: oss << "unknown_step_" << static_cast<int>(step); break;
    }
    oss << "_dim" << dim_n << ".raw";
    std::string filename = oss.str();
    
    std::ofstream outfile(filename, std::ios::binary | std::ios::out);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    outfile.close();
    
    // Calculate and print statistics
    DataStats stats = calculateStats(data, filename);
    
    std::cout << "Saved intermediate data to " << filename 
              << " (Size: " << data.size() << ")" << std::endl;
}

const std::vector<float>& WaveletNoise::getNoiseCoefficients() const {
    return noiseCoefficients;
}

int WaveletNoise::getTileSize() const {
    return tileSizeN;
}