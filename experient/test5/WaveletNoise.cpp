#include "WaveletNoise.h"
#include <cmath>     // For ceil, abs
#include <iostream>  // For debugging
#include <numeric>   // For std::iota (if needed for testing)
#include <algorithm> // For std::fill
#include <sstream>   // For std::ostringstream
#include <fstream>   // For std::ofstream

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
        //tileSizeN++; // Paper: "tile size must be even" for N=R-Rdsus
        // However, the C code `if (n%2) n++;` is inside GenerateNoiseTile,
        // suggesting 'n' is a local copy. We'll enforce it here or adjust later.
        std::cerr << "Warning: Tile size should ideally be even for exact paper replication of N=R-Rdsus properties." << std::endl;
    }
}

WaveletNoise::~WaveletNoise() {}

int WaveletNoise::Mod(int x, int n) const {
    int m = x % n;
    return (m < 0) ? m + n : m;
}

// Based on paper's Downsample
void WaveletNoise::downsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride) {
    // 'a_ptr_center' points to the logical center of A_COEFFS (A_COEFFS[ARAD])
    const float* a_ptr_center = &A_COEFFS[ARAD]; // a in C code
    std::fill(to.begin(), to.end(), 0.0f); // Ensure 'to' is zeroed out if it's reused

    for (int i_out = 0; i_out < n / 2; ++i_out) {
        float sum = 0.0f;
        for (int k_filt_offset = -ARAD; k_filt_offset < ARAD; ++k_filt_offset) {
            int k_in_signal = 2 * i_out + k_filt_offset;
            sum += a_ptr_center[k_filt_offset] * from[Mod(k_in_signal, n) * stride];
        }
        to[i_out * stride] = sum;
    }
}

// Based on paper's Upsample
void WaveletNoise::upsample1D(const std::vector<float>& from_coarse, std::vector<float>& to_fine, int n_fine, int stride) {
    // 'p_ptr_center' points to P_COEFFS[2] (logical p[0] in C code)
    const float* p_ptr_center = &P_COEFFS[2]; // p in C code
    std::fill(to_fine.begin(), to_fine.end(), 0.0f);

    int n_coarse = n_fine / 2; // Size of from_coarse array for Mod operation

    for (int i_fine = 0; i_fine < n_fine; ++i_fine) {
        float sum = 0.0f;
        // Loop for k in C code: for (int k=i_fine/2; k<=i_fine/2+1; k++)
        // This means two contributions from 'from_coarse'
        for (int k_coarse_loop_idx_offset = 0; k_coarse_loop_idx_offset < 2; ++k_coarse_loop_idx_offset) {
            int k_coarse = i_fine / 2 + k_coarse_loop_idx_offset;
            int p_filt_idx = i_fine - 2 * k_coarse; // Index for p_ptr_center. Ranges -2 to 1.
                                                  // P_COEFFS indices: p_ptr_center[0] = P_COEFFS[2]
                                                  // p_ptr_center[-1]= P_COEFFS[1]
                                                  // p_ptr_center[-2]= P_COEFFS[0]
                                                  // p_ptr_center[1] = P_COEFFS[3]
            if (p_filt_idx >= -2 && p_filt_idx <= 1) { // Check if valid for our P_COEFFS mapping
                 // This direct indexing of p_ptr_center assumes p_filt_idx is relative to it.
                sum += p_ptr_center[p_filt_idx] * from_coarse[Mod(k_coarse, n_coarse) * stride];
            }
        }
        to_fine[i_fine * stride] = sum;
    }
}


void WaveletNoise::generateNoiseTile() {
    // This will implement the 3D version from Appendix 1
    generateNoiseTile3D();
}

void WaveletNoise::generateNoiseTile2D() {
    int n = tileSizeN;
    if (n % 2) n++;

    int num_elements = n * n; // 2D: n*n
    noiseCoefficients.assign(num_elements, 0.0f);
    std::vector<float> temp1(num_elements);
    std::vector<float> temp2(num_elements);

    std::vector<float> r_initial_copy(num_elements);

    // Step 1: Fill with random numbers
    for (int i = 0; i < num_elements; ++i) {
        r_initial_copy[i] = gaussianDist(rng);
    }

    std::vector<float> current_data_for_filtering = r_initial_copy;

    std::vector<float> line_from(n);
    std::vector<float> line_to_ds(n / 2);
    std::vector<float> line_to_us(n);

    // --- Process along x-axis ---
    std::fill(temp2.begin(), temp2.end(), 0.0f);
    for (int iy = 0; iy < n; ++iy) {
        int base_idx = iy * n;
        for(int ix_line = 0; ix_line < n; ++ix_line) line_from[ix_line] = current_data_for_filtering[base_idx + ix_line];
        downsample1D(line_from, line_to_ds, n, 1);
        upsample1D(line_to_ds, line_to_us, n, 1);
        for(int ix_line = 0; ix_line < n; ++ix_line) temp2[base_idx + ix_line] = line_to_us[ix_line];
    }
    current_data_for_filtering = temp2;

    // --- Process along y-axis ---
    std::fill(temp1.begin(), temp1.end(), 0.0f);
    for (int ix = 0; ix < n; ++ix) {
        int base_idx = ix;
        for(int iy_line = 0; iy_line < n; ++iy_line) line_from[iy_line] = current_data_for_filtering[base_idx + iy_line * n];
        downsample1D(line_from, line_to_ds, n, 1);
        upsample1D(line_to_ds, line_to_us, n, 1);
        for(int iy_line = 0; iy_line < n; ++iy_line) temp1[base_idx + iy_line * n] = line_to_us[iy_line];
    }
    // temp1 現在儲存了 R_ds_us_xy

    // Step 4: Subtract to get N_coeffs
    noiseCoefficients.resize(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] = r_initial_copy[i] - temp1[i]; // 注意這裡是 temp1
    }

    // --- Avoid even/odd variance difference (2D version) ---
    int offset = n / 2;
    if (offset % 2 == 0) offset++;

    temp1.assign(num_elements, 0.0f);
    std::vector<float> n_coeffs_copy_for_shift = noiseCoefficients;
    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            int current_idx = ix + iy * n;
            int offset_idx = Mod(ix + offset, n) +
                             Mod(iy + offset, n) * n;
            temp1[current_idx] = n_coeffs_copy_for_shift[offset_idx];
        }
    }
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] += temp1[i];
    }
}

void WaveletNoise::generateNoiseTile3D() {
    int n = tileSizeN;
    if (n % 2) n++;

    int num_elements = n * n * n;
    noiseCoefficients.assign(num_elements, 0.0f);
    std::vector<float> temp1(num_elements);
    std::vector<float> temp2(num_elements); // Will hold R_ds_us results per axis

    std::vector<float> r_initial_copy(num_elements); // To keep original R

    // Step 1: Fill the tile with random numbers
    for (int i = 0; i < num_elements; ++i) {
        r_initial_copy[i] = gaussianDist(rng);
    }
    saveIntermediateData(r_initial_copy, n, DebugStep::R_INITIAL);

    // Initialize temp2 with R_initial for the first axis processing
    // (or noiseCoefficients if we plan to modify it in place for R_ds_us)
    // The C code seems to apply filters to 'noise' (which is R_initial) and stores in temp1/temp2
    // Let's use r_initial_copy as 'from' and store results in temp2/temp1
    std::vector<float> current_data_for_filtering = r_initial_copy;


    std::vector<float> line_from(n);
    std::vector<float> line_to_ds(n / 2);
    std::vector<float> line_to_us(n);

    // Process along x-axis
    std::fill(temp2.begin(), temp2.end(), 0.0f); // temp2 will store R_ds_us_x
    for (int iy = 0; iy < n; ++iy) {
        for (int iz = 0; iz < n; ++iz) {
            int base_idx = iy * n + iz * n * n;
            for(int ix_line=0; ix_line<n; ++ix_line) line_from[ix_line] = current_data_for_filtering[base_idx + ix_line];
            downsample1D(line_from, line_to_ds, n, 1);
            upsample1D(line_to_ds, line_to_us, n, 1);
            for(int ix_line=0; ix_line<n; ++ix_line) temp2[base_idx + ix_line] = line_to_us[ix_line];
        }
    }
    saveIntermediateData(temp2, n, DebugStep::AFTER_X_FILTER);
    current_data_for_filtering = temp2; // Next stage filters this

    // Process along y-axis
    std::fill(temp1.begin(), temp1.end(), 0.0f); // temp1 will store R_ds_us_xy
    for (int ix = 0; ix < n; ++ix) {
        for (int iz = 0; iz < n; ++iz) {
            int base_idx = ix + iz * n * n;
            for(int iy_line=0; iy_line<n; ++iy_line) line_from[iy_line] = current_data_for_filtering[base_idx + iy_line * n];
            downsample1D(line_from, line_to_ds, n, 1);
            upsample1D(line_to_ds, line_to_us, n, 1);
            for(int iy_line=0; iy_line<n; ++iy_line) temp1[base_idx + iy_line*n] = line_to_us[iy_line];
        }
    }
    saveIntermediateData(temp1, n, DebugStep::AFTER_Y_FILTER);
    current_data_for_filtering = temp1; // Next stage filters this

    // Process along z-axis
    std::fill(temp2.begin(), temp2.end(), 0.0f); // temp2 will store R_ds_us_xyz (R_downarrow_uparrow)
    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            int base_idx = ix + iy * n;
            for(int iz_line=0; iz_line<n; ++iz_line) line_from[iz_line] = current_data_for_filtering[base_idx + iz_line * n * n];
            downsample1D(line_from, line_to_ds, n, 1);
            upsample1D(line_to_ds, line_to_us, n, 1);
            for(int iz_line=0; iz_line<n; ++iz_line) temp2[base_idx + iz_line * n * n] = line_to_us[iz_line];
        }
    }
    saveIntermediateData(temp2, n, DebugStep::AFTER_Z_FILTER); // This is R_downarrow_uparrow

    // Step 4: Subtract to get N_coeffs = R_initial - R_downarrow_uparrow
    noiseCoefficients.resize(num_elements); // Ensure correct size
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] = r_initial_copy[i] - temp2[i];
    }
    saveIntermediateData(noiseCoefficients, n, DebugStep::N_COEFFS_PRE_CORRECTION);

    // Avoid even/odd variance difference
    int offset = n / 2;
    if (offset % 2 == 0) offset++;

    temp1.assign(num_elements, 0.0f); // Reuse temp1 for shifted N_coeffs
    std::vector<float> n_coeffs_copy_for_shift = noiseCoefficients; // Keep original N before adding shifted
    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            for (int iz = 0; iz < n; ++iz) {
                int current_idx = ix + iy * n + iz * n * n;
                int offset_idx = Mod(ix + offset, n) +
                                 Mod(iy + offset, n) * n +
                                 Mod(iz + offset, n) * n * n;
                temp1[current_idx] = n_coeffs_copy_for_shift[offset_idx]; // Shift from original N
            }
        }
    }
    for (int i = 0; i < num_elements; ++i) {
        noiseCoefficients[i] += temp1[i]; // Add shifted version to original N
    }
    saveIntermediateData(noiseCoefficients, n, DebugStep::N_COEFFS_FINAL);
}


float WaveletNoise::calculateTotalEnergy(const std::vector<float>& data) const {
    double sum_sq = 0.0; // Use double for sum to maintain precision
    for (float val : data) {
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
    }
    return static_cast<float>(sum_sq);
}

void WaveletNoise::saveIntermediateData(const std::vector<float>& data, int dim_n, DebugStep step, const std::string& base_filename) const {
    if (data.empty()) {
        std::cerr << "Data is empty for step " << static_cast<int>(step) << ", not saving." << std::endl;
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

    float energy = calculateTotalEnergy(data);
    std::cout << "Saved intermediate data to " << filename << " (Size: " << data.size() << ", Energy: " << energy << ")" << std::endl;
}

// Evaluate 3D noise (based on WNoise from Appendix 2)
float WaveletNoise::evaluate3D(const float p_world[3]) const {
    // p_world are the world-space coordinates. For a noise band N(2^b * x),
    // the input to this evaluation (which uses N_coeffs for N(s)) should be s = 2^b * x.
    // The paper's WNoise takes 'p[3]' which seems to be already scaled (e.g. q[i] = 2*p[i]*pow(2, firstBand+b) in WMultibandNoise)
    // The basis functions are B(2s-i) in construction of N.
    // For evaluation N(s_eval) = sum n_i B(2*s_eval - i).
    // Let q = 2*s_eval. Then N(s_eval) = sum n_i B(q - i).
    // The C code for WNoise implies p is 'q'.
    // `mid[i]=ceil(p[i]-0.5); t=mid[i]-(p[i]-0.5);` means p[i] is the evaluation point for B(t).

    float result = 0.0f;
    int f[3], c[3], mid[3]; // filter offsets, coefficient indices, mid-point coefficient indices
    float w[3][3]; // basis function weights [dim][offset_idx+1]

    int n_current_tile_dim = static_cast<int>(std::round(std::cbrt(noiseCoefficients.size())));
    if (n_current_tile_dim == 0 && noiseCoefficients.empty()) return 0.0f; // Handle empty tile
    if (n_current_tile_dim == 0 && !noiseCoefficients.empty()) {
        std::cerr << "Error: Tile data exists but dimension is zero." << std::endl;
        return 0.0f;
    }


    for (int i = 0; i < 3; ++i) {
        mid[i] = static_cast<int>(std::ceil(p_world[i] - 0.5f));
        float t_basis = mid[i] - (p_world[i] - 0.5f); // t for B-spline eval, relative to center of support of B(x-mid[i])
        // B(t_basis) where t_basis is dist from center of B_mid[i]'s support to p_world[i]
        // Weights for B(p- (mid+f))
        // w[i][0] for f=-1, w[i][1] for f=0, w[i][2] for f=1
        w[i][0] = t_basis * t_basis / 2.0f;
        w[i][2] = (1.0f - t_basis) * (1.0f - t_basis) / 2.0f;
        w[i][1] = 1.0f - w[i][0] - w[i][2];
    }

    for (f[2] = -1; f[2] <= 1; ++f[2]) {
        for (f[1] = -1; f[1] <= 1; ++f[1]) {
            for (f[0] = -1; f[0] <= 1; ++f[0]) {
                float weight = 1.0f;
                int current_coeff_flat_idx = 0;
                int dim_stride = 1;
                for (int i = 0; i < 3; ++i) { // 0:x, 1:y, 2:z
                    c[i] = Mod(mid[i] + f[i], n_current_tile_dim);
                    weight *= w[i][f[i] + 1]; // f[i]+1 maps -1,0,1 to 0,1,2 for w array
                    current_coeff_flat_idx += c[i] * dim_stride;
                    dim_stride *= n_current_tile_dim;
                }
                if (current_coeff_flat_idx >=0 && current_coeff_flat_idx < noiseCoefficients.size()){
                     result += weight * noiseCoefficients[current_coeff_flat_idx];
                } else {
                    // This case should ideally not happen if Mod and n_current_tile_dim are correct
                    // std::cerr << "Warning: coefficient index out of bounds in eval3D" << std::endl;
                }
            }
        }
    }
    return result;
}


// Placeholder for 2D evaluation
float WaveletNoise::evaluate2D(float u, float v) const {
    if (noiseCoefficients.empty()) return 0.0f;
    int n_current_tile_dim = static_cast<int>(std::round(std::sqrt(noiseCoefficients.size())));
     if (n_current_tile_dim == 0) return 0.0f;


    // Map normalized u,v to the p = 2s scale for B(p-i) evaluation
    float p_eval[2];
    p_eval[0] = 2.0f * u * n_current_tile_dim; // s = u * tile_size_in_coeff_units
    p_eval[1] = 2.0f * v * n_current_tile_dim; // tile_size_in_coeff_units is n_current_tile_dim

    float result = 0.0f;
    int f[2], c[2], mid[2];
    float w[2][3];

    for (int i = 0; i < 2; ++i) {
        mid[i] = static_cast<int>(std::ceil(p_eval[i] - 0.5f));
        float t_basis = mid[i] - (p_eval[i] - 0.5f);
        w[i][0] = t_basis * t_basis / 2.0f;
        w[i][2] = (1.0f - t_basis) * (1.0f - t_basis) / 2.0f;
        w[i][1] = 1.0f - w[i][0] - w[i][2];
    }

    for (f[1] = -1; f[1] <= 1; ++f[1]) { // y loop
        for (f[0] = -1; f[0] <= 1; ++f[0]) { // x loop
            float weight = 1.0f;
            int current_coeff_flat_idx = 0;
            int dim_stride = 1;

            c[0] = Mod(mid[0] + f[0], n_current_tile_dim); // x coeff index
            c[1] = Mod(mid[1] + f[1], n_current_tile_dim); // y coeff index

            weight = w[0][f[0] + 1] * w[1][f[1] + 1];
            
            current_coeff_flat_idx = c[0] + c[1] * n_current_tile_dim; // For 2D tile (ix + iy * size_x)

            if (current_coeff_flat_idx >=0 && current_coeff_flat_idx < noiseCoefficients.size()){
                 result += weight * noiseCoefficients[current_coeff_flat_idx];
            }
        }
    }
    return result;
}

float WaveletNoise::evaluate3DProjected(const float p[3], const float normal[3]) const {
    if (noiseCoefficients.empty()) return 0.0f;
    
    int n_current_tile_dim = static_cast<int>(std::round(std::cbrt(noiseCoefficients.size())));
    if (n_current_tile_dim == 0) return 0.0f;

    float result = 0.0f;
    int c[3]; // 噪聲係數位置
    
    // --- Bound the support of the basis functions for this projection direction ---
    // 這是論文中的一個關鍵優化，根據法線方向計算出可能影響當前點的係數邊界框
    int min_c[3], max_c[3];
    for (int i = 0; i < 3; ++i) {
        // 這個公式直接來自論文附錄，是預先計算好的非對稱基函數的支撐範圍
        float support = 3.0f * std::abs(normal[i]) + 3.0f * std::sqrt((1.0f - normal[i] * normal[i]) / 2.0f);
        min_c[i] = static_cast<int>(std::ceil(p[i] - support));
        max_c[i] = static_cast<int>(std::floor(p[i] + support));
    }

    // --- Loop over the noise coefficients within the bound ---
    for (c[2] = min_c[2]; c[2] <= max_c[2]; ++c[2]) {
        for (c[1] = min_c[1]; c[1] <= max_c[1]; ++c[1]) {
            for (c[0] = min_c[0]; c[0] <= max_c[0]; ++c[0]) {
                // --- Dot the normal with the vector from coefficient c to point p ---
                float dot = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    dot += normal[i] * (p[i] - c[i]);
                }

                float weight = 1.0f;
                // --- Evaluate the basis function (approximating the integral) ---
                // 這是論文中最核心的近似技巧
                for (int i = 0; i < 3; ++i) {
                    // 局部座標 t 是在一個沿法線方向移動和縮放過的 B-Spline 基函數中計算的
                    float t = (static_cast<float>(c[i]) + normal[i] * dot / 2.0f) - (p[i] - 1.5f);
                    
                    // 標準二次 B-Spline 求值
                    if (t <= 0.0f || t >= 3.0f) {
                        weight = 0.0f;
                        break; // 提前退出內層循環
                    } else if (t < 1.0f) {
                        weight *= (t * t / 2.0f);
                    } else if (t < 2.0f) {
                        float t1 = t - 1.0f;
                        float t2 = 2.0f - t;
                        weight *= (1.0f - (t1 * t1 + t2 * t2) / 2.0f);
                    } else { // t < 3.0f
                        float t3 = 3.0f - t;
                        weight *= (t3 * t3 / 2.0f);
                    }
                }

                if (weight > 1e-6) { // 優化：如果權重太小就忽略
                    int idx = Mod(c[0], n_current_tile_dim) +
                              Mod(c[1], n_current_tile_dim) * n_current_tile_dim +
                              Mod(c[2], n_current_tile_dim) * n_current_tile_dim * n_current_tile_dim;
                    result += weight * noiseCoefficients[idx];
                }
            }
        }
    }
    return result;
}

const std::vector<float>& WaveletNoise::getNoiseCoefficients() const {
    return noiseCoefficients;
}

int WaveletNoise::getTileSize() const {
    return tileSizeN;
}