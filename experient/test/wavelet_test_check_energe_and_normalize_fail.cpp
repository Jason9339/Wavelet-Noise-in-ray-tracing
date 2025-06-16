// wavelet_noise_enhanced.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <complex>
#include <cstring>
#include <numeric>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===== BMP file output tools =====
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t type = 0x4D42;
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
    if (data.empty() || data[0].empty()) {
        std::cerr << "Error: Empty data for saveBMP: " << filename << std::endl;
        return;
    }
    int width = data[0].size();
    int height = data.size();

    BMPHeader header;
    header.width = width;
    header.height = height;
    int row_stride = (width * 3 + 3) & ~3;
    header.image_size = row_stride * height;
    header.size = header.offset + header.image_size;

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<char*>(&header), sizeof(header));

    std::vector<uint8_t> row_buffer(row_stride);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            float val = data[y][x];
            if (std::isnan(val) || std::isinf(val)) val = 0.0f;
            // Normalize from [-1, 1] to [0, 255]
            uint8_t value = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (val + 1.0f) * 127.5f)));
            row_buffer[x * 3 + 0] = value;
            row_buffer[x * 3 + 1] = value;
            row_buffer[x * 3 + 2] = value;
        }
        for (int i = width * 3; i < row_stride; ++i) {
            row_buffer[i] = 0;
        }
        file.write(reinterpret_cast<char*>(row_buffer.data()), row_stride);
    }
    file.close();
    std::cout << "Saved BMP: " << filename << std::endl;
}

void saveDataAsCSV(const std::string& filename, const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing CSV: " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            file << data[i][j] << (j == data[i].size() - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    std::cout << "Saved CSV data: " << filename << std::endl;
}

// ===== Math utility functions =====
inline int Mod(int x, int n) {
    if (n == 0) { std::cerr << "Error: Modulo by zero." << std::endl; return 0; }
    if (n < 0) n = -n;
    int m = x % n;
    return (m < 0) ? m + n : m;
}

float gaussianNoise() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

// ===== FFT Implementation =====
using Complex = std::complex<float>;

void fft1D(std::vector<Complex>& a, bool invert) {
    int n = a.size();
    if (n <= 1) return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        Complex wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int j = 0; j < len / 2; j++) {
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (Complex& x : a) x /= n;
    }
}

void fft2D(const std::vector<std::vector<float>>& input, std::vector<std::vector<Complex>>& output) {
    if (input.empty() || input[0].empty()) {
        std::cerr << "FFT Error: Input data is empty." << std::endl;
        output.clear();
        return;
    }
    int height = input.size();
    int width = input[0].size();
    if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0 || height == 0 || width == 0) {
        std::cerr << "FFT Error: Image dimensions must be powers of 2." << std::endl;
        output.assign(height, std::vector<Complex>(width, 0));
        return;
    }
    output.assign(height, std::vector<Complex>(width));
    for(int i=0; i<height; ++i) {
        for(int j=0; j<width; ++j) {
            output[i][j] = std::isfinite(input[i][j]) ? input[i][j] : 0.0f;
        }
    }
    for (int i = 0; i < height; i++) fft1D(output[i], false);
    std::vector<Complex> col(height);
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) col[i] = output[i][j];
        fft1D(col, false);
        for (int i = 0; i < height; i++) output[i][j] = col[i];
    }
}

void saveSpectrum(const std::string& filename_bmp, const std::vector<std::vector<Complex>>& spectrum) {
    if (spectrum.empty() || spectrum[0].empty()) {
        std::cerr << "Error: Empty spectrum data." << std::endl;
        return;
    }
    int height = spectrum.size();
    int width = spectrum[0].size();

    std::vector<std::vector<float>> magnitudes(height, std::vector<float>(width));
    float maxMag = 0.0f;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = std::abs(spectrum[i][j]);
            magnitudes[i][j] = std::isfinite(mag) ? mag : 0.0f;
            if (magnitudes[i][j] > maxMag) maxMag = magnitudes[i][j];
        }
    }

    // Shift spectrum to center DC component
    std::vector<std::vector<float>> shifted_magnitudes(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            shifted_magnitudes[i][j] = magnitudes[(i + height / 2) % height][(j + width / 2) % width];
        }
    }
    
    // Save raw magnitude data
    std::string filename_csv = filename_bmp;
    size_t dot_pos = filename_csv.rfind(".bmp");
    if (dot_pos != std::string::npos) filename_csv.replace(dot_pos, 4, "_magnitude.csv");
    else filename_csv += "_magnitude.csv";
    saveDataAsCSV(filename_csv, shifted_magnitudes);

    // Log scale for visualization
    std::vector<std::vector<float>> log_magnitudes_for_bmp(height, std::vector<float>(width));
    float min_log_m = std::numeric_limits<float>::max();
    float max_log_m = std::numeric_limits<float>::lowest();
    const float FACTOR = 10000.0f;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag_val = shifted_magnitudes[i][j];
            float current_log_mag;
            if (maxMag > 1e-9f) {
                current_log_mag = std::log(1.0f + mag_val / maxMag * FACTOR);
            } else {
                current_log_mag = 0.0f;
            }
            log_magnitudes_for_bmp[i][j] = current_log_mag;

            if (std::isfinite(current_log_mag)) {
                if (current_log_mag < min_log_m) min_log_m = current_log_mag;
                if (current_log_mag > max_log_m) max_log_m = current_log_mag;
            }
        }
    }
    
    if (std::abs(max_log_m - min_log_m) < 1e-6f) {
        max_log_m = min_log_m + 1.0f;
    }

    std::vector<std::vector<float>> output_img(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (std::isfinite(log_magnitudes_for_bmp[i][j]) && max_log_m - min_log_m > 1e-6f) {
                output_img[i][j] = 2.0f * (log_magnitudes_for_bmp[i][j] - min_log_m) / (max_log_m - min_log_m) - 1.0f;
            } else {
                output_img[i][j] = -1.0f;
            }
        }
    }
    saveBMP(filename_bmp, output_img);
}

// ===== Statistical analysis =====
struct DataStats {
    float avg = 0.0f;
    float var = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    long long count_nan_inf = 0;
    float energy = 0.0f;  // Sum of squares
};

DataStats calculateStats(const float* data, long long total_size, const std::string& name) {
    DataStats stats;
    if (total_size == 0) {
        std::cerr << "Warning: Zero-sized data: " << name << std::endl;
        return stats;
    }
    
    double sum = 0.0, sum_sq = 0.0;
    long long valid_count = 0;
    
    for (long long i = 0; i < total_size; ++i) {
        float val = data[i];
        if (std::isnan(val) || std::isinf(val)) {
            stats.count_nan_inf++;
            continue;
        }
        valid_count++;
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
    }
    
    if (valid_count > 0) {
        stats.avg = static_cast<float>(sum / valid_count);
        stats.var = static_cast<float>((sum_sq / valid_count) - static_cast<double>(stats.avg) * stats.avg);
        stats.energy = static_cast<float>(sum_sq);
    } else {
        std::cerr << "Warning: No valid data for stats: " << name << std::endl;
        stats.min_val = 0;
        stats.max_val = 0;
    }
    
    std::cout << name << " stats: avg=" << stats.avg << " var=" << stats.var
              << " stddev=" << (stats.var > 0 ? sqrt(stats.var) : 0)
              << " min=" << stats.min_val << " max=" << stats.max_val
              << " energy=" << stats.energy;
    if (stats.count_nan_inf > 0) std::cout << " (NaN/Inf: " << stats.count_nan_inf << ")";
    std::cout << std::endl;
    
    return stats;
}

// Function to analyze frequency content
void analyzeFrequencyContent(const std::vector<std::vector<Complex>>& spectrum, const std::string& name) {
    if (spectrum.empty() || spectrum[0].empty()) return;
    
    int height = spectrum.size();
    int width = spectrum[0].size();
    int cy = height / 2;
    int cx = width / 2;
    
    // Analyze energy in different frequency bands
    float dc_energy = 0.0f;
    float low_freq_energy = 0.0f;   // Within radius 8
    float mid_freq_energy = 0.0f;   // Radius 8-32
    float high_freq_energy = 0.0f;  // Radius > 32
    float total_energy = 0.0f;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = std::abs(spectrum[i][j]);
            float energy = mag * mag;
            total_energy += energy;
            
            // Calculate distance from DC (after shifting)
            int dy = (i + cy) % height - cy;
            int dx = (j + cx) % width - cx;
            float dist = std::sqrt(dy*dy + dx*dx);
            
            if (dist < 1.0f) {
                dc_energy += energy;
            } else if (dist < 8.0f) {
                low_freq_energy += energy;
            } else if (dist < 32.0f) {
                mid_freq_energy += energy;
            } else {
                high_freq_energy += energy;
            }
        }
    }
    
    std::cout << "\nFrequency Analysis for " << name << ":" << std::endl;
    std::cout << "  DC component: " << (dc_energy/total_energy*100) << "%" << std::endl;
    std::cout << "  Low freq (r<8): " << (low_freq_energy/total_energy*100) << "%" << std::endl;
    std::cout << "  Mid freq (8≤r<32): " << (mid_freq_energy/total_energy*100) << "%" << std::endl;
    std::cout << "  High freq (r≥32): " << (high_freq_energy/total_energy*100) << "%" << std::endl;
    std::cout << "  Total energy: " << total_energy << std::endl;
}

void saveTileSlice(const std::string& filename_prefix, int slice_z, const float* tile_data, int n) {
    if (!tile_data || n <= 0 || slice_z < 0 || slice_z >= n) {
        std::cerr << "Error: Invalid args for saveTileSlice." << std::endl;
        return;
    }
    
    std::vector<std::vector<float>> slice_image(n, std::vector<float>(n));
    float current_min_val = std::numeric_limits<float>::max();
    float current_max_val = std::numeric_limits<float>::lowest();

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            long long index = static_cast<long long>(x) + static_cast<long long>(y) * n + static_cast<long long>(slice_z) * n * n;
            float val = tile_data[index];
            slice_image[y][x] = val;
            if (std::isfinite(val)) {
                if (val < current_min_val) current_min_val = val;
                if (val > current_max_val) current_max_val = val;
            }
        }
    }
    
    // Normalize for visualization
    if ((current_max_val - current_min_val > 1e-6f) && std::isfinite(current_min_val) && std::isfinite(current_max_val)) {
        for(int y=0; y<n; ++y) {
            for(int x=0; x<n; ++x) {
                slice_image[y][x] = std::isfinite(slice_image[y][x]) ? 
                    2.0f * (slice_image[y][x] - current_min_val) / (current_max_val - current_min_val) - 1.0f : 0.0f;
            }
        }
    }

    saveBMP(filename_prefix + "_slice_z" + std::to_string(slice_z) + ".bmp", slice_image);
    
    // FFT analysis
    if (n > 0 && (n & (n - 1)) == 0) {
        std::vector<std::vector<Complex>> spectrum_data;
        fft2D(slice_image, spectrum_data);
        saveSpectrum(filename_prefix + "_slice_z" + std::to_string(slice_z) + "_spectrum.bmp", spectrum_data);
        
        // Analyze frequency content
        analyzeFrequencyContent(spectrum_data, filename_prefix + "_slice_z" + std::to_string(slice_z));
    }
}

// ===== Wavelet Noise Implementation =====
class WaveletNoise {
public:
    static float* noiseTileData;
    static int noiseTileSize;
    static constexpr int ARAD = 16;

    static void GenerateNoiseTile(int n_target, int /*olap*/ = 0) {
        if (n_target <= 0) {
            std::cerr << "Error: Invalid tile size." << std::endl;
            return;
        }
        
        int n = n_target;
        if (n % 2) n++;  // Ensure even size
        noiseTileSize = n;
        long long num_elements = static_cast<long long>(n) * n * n;
        size_t sz_bytes = num_elements * sizeof(float);

        float* r_initial  = (float*)malloc(sz_bytes);
        float* temp1      = (float*)malloc(sz_bytes);
        float* r_down_up  = (float*)malloc(sz_bytes);
        float* n_final    = (float*)malloc(sz_bytes);

        if (!r_initial || !temp1 || !r_down_up || !n_final) {
            std::cerr << "Error: Memory allocation failed." << std::endl;
            free(r_initial); free(temp1); free(r_down_up); free(n_final);
            noiseTileData = nullptr; noiseTileSize = 0;
            return;
        }

        std::cout << "\n=== WAVELET NOISE GENERATION PROCESS ===" << std::endl;
        std::cout << "Tile size: " << n << "x" << n << "x" << n << std::endl;

        // Step 1: Fill with Gaussian noise
        std::cout << "\nStep 1: Filling with Gaussian noise..." << std::endl;
        for (long long i = 0; i < num_elements; i++) {
            r_initial[i] = gaussianNoise();
        }
        DataStats stats_r_initial = calculateStats(r_initial, num_elements, "R_initial");
        if ((n&(n-1))==0) saveTileSlice("debug_R_initial", n / 2, r_initial, n);

        // Steps 2 & 3: Downsample and Upsample
        std::cout << "\nSteps 2 & 3: Downsampling and Upsampling..." << std::endl;
        
        // Copy r_initial to r_down_up first
        memcpy(r_down_up, r_initial, sz_bytes);
        
        // X-direction
        std::cout << "  Processing X direction..." << std::endl;
        for (int iy = 0; iy < n; iy++) {
            for (int iz = 0; iz < n; iz++) {
                long long base_idx = static_cast<long long>(iy) * n + static_cast<long long>(iz) * n * n;
                Downsample(&r_down_up[base_idx], &temp1[base_idx], n, 1);
                Upsample(&temp1[base_idx], &r_down_up[base_idx], n, 1);
            }
        }
        
        // Y-direction
        std::cout << "  Processing Y direction..." << std::endl;
        for (int ix = 0; ix < n; ix++) {
            for (int iz = 0; iz < n; iz++) {
                long long base_idx = static_cast<long long>(ix) + static_cast<long long>(iz) * n * n;
                Downsample(&r_down_up[base_idx], &temp1[base_idx], n, n);
                Upsample(&temp1[base_idx], &r_down_up[base_idx], n, n);
            }
        }
        
        // Z-direction
        std::cout << "  Processing Z direction..." << std::endl;
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                long long base_idx = static_cast<long long>(ix) + static_cast<long long>(iy) * n;
                Downsample(&r_down_up[base_idx], &temp1[base_idx], n, n * n);
                Upsample(&temp1[base_idx], &r_down_up[base_idx], n, n * n);
            }
        }
        
        DataStats stats_r_down_up = calculateStats(r_down_up, num_elements, "R_down_up");
        if ((n&(n-1))==0) saveTileSlice("debug_R_down_up", n / 2, r_down_up, n);
        
        // Step 4: Subtract to get high-frequency component
        std::cout << "\nStep 4: Computing N_intermediate = R_initial - R_down_up..." << std::endl;
        for (long long i = 0; i < num_elements; i++) {
            n_final[i] = r_initial[i] - r_down_up[i];
        }
        DataStats stats_n_intermediate = calculateStats(n_final, num_elements, "N_intermediate");
        if ((n&(n-1))==0) saveTileSlice("debug_N_intermediate", n / 2, n_final, n);

        // Step 5: Add offset to avoid even/odd variance difference
        std::cout << "\nStep 5: Adding offset for variance correction..." << std::endl;
        int offset_val = n / 2;
        if (offset_val % 2 == 0) offset_val++;
        
        // Use temp1 for offset version
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                for (int iz = 0; iz < n; iz++) {
                    long long target_idx = static_cast<long long>(ix) + 
                                         static_cast<long long>(iy) * n + 
                                         static_cast<long long>(iz) * n * n;
                    long long src_idx = static_cast<long long>(Mod(ix + offset_val, n)) +
                                      static_cast<long long>(Mod(iy + offset_val, n)) * n +
                                      static_cast<long long>(Mod(iz + offset_val, n)) * n * n;
                    temp1[target_idx] = n_final[src_idx];
                }
            }
        }
        
        for (long long i = 0; i < num_elements; i++) {
            n_final[i] += temp1[i];
        }
        
        DataStats stats_n_final = calculateStats(n_final, num_elements, "N_final");
        if ((n&(n-1))==0) saveTileSlice("debug_N_final", n / 2, n_final, n);
        
        // Analyze the energy ratio
        std::cout << "\n=== ENERGY ANALYSIS ===" << std::endl;
        std::cout << "Energy ratio (R_down_up / R_initial): " << stats_r_down_up.energy / stats_r_initial.energy << std::endl;
        std::cout << "Energy ratio (N_final / R_initial): " << stats_n_final.energy / stats_r_initial.energy << std::endl;
        
        free(r_initial);
        free(temp1);
        free(r_down_up);
        noiseTileData = n_final;
        
        std::cout << "\n=== NOISE TILE GENERATION COMPLETE ===" << std::endl;
    }

    static void Downsample(const float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) {
            std::cerr << "Error: Invalid args to Downsample." << std::endl;
            return;
        }
        
        // Analysis filter coefficients (low-pass)
        float aCoeffs[2*ARAD] = {
            0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
            -0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
            0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
            0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
        };
        
        const float* a = &aCoeffs[ARAD];
        
        for (int i = 0; i < n / 2; i++) {
            long long to_idx = static_cast<long long>(i) * stride;
            to[to_idx] = 0.0f;
            
            for (int k = -ARAD; k < ARAD; k++) {
                int from_idx = Mod(2*i + k, n);
                long long from_element_idx = static_cast<long long>(from_idx) * stride;
                to[to_idx] += a[k] * from[from_element_idx];
            }
        }
    }

    static void Upsample(const float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) {
            std::cerr << "Error: Invalid args to Upsample." << std::endl;
            return;
        }
        
        // Refinement coefficients
        float pCoeffs[4] = { 0.25f, 0.75f, 0.75f, 0.25f };
        
        for (int i = 0; i < n; i++) {
            long long to_idx = static_cast<long long>(i) * stride;
            to[to_idx] = 0.0f;
            
            for (int k = 0; k <= 1; k++) {
                int from_idx = i/2 - k;
                int p_idx = i - 2*from_idx;
                
                if (p_idx >= 0 && p_idx < 4) {
                    long long from_element_idx = static_cast<long long>(Mod(from_idx, n/2)) * stride;
                    to[to_idx] += pCoeffs[p_idx] * from[from_element_idx];
                }
            }
        }
    }

    static float WNoise(float p[3]) {
        int n = noiseTileSize;
        if (n == 0 || !noiseTileData) return 0.0f;
        
        int i_dim, f_offset[3], mid_cell[3];
        float w_basis[3][3], t_param, result = 0.0f;

        for (i_dim = 0; i_dim < 3; i_dim++) {
            mid_cell[i_dim] = static_cast<int>(std::ceil(p[i_dim] - 0.5f));
            t_param = mid_cell[i_dim] - (p[i_dim] - 0.5f);
            w_basis[i_dim][0] = t_param * t_param / 2.0f;
            w_basis[i_dim][2] = (1.0f - t_param) * (1.0f - t_param) / 2.0f;
            w_basis[i_dim][1] = 1.0f - w_basis[i_dim][0] - w_basis[i_dim][2];
        }

        for (f_offset[2] = -1; f_offset[2] <= 1; f_offset[2]++) {
            for (f_offset[1] = -1; f_offset[1] <= 1; f_offset[1]++) {
                for (f_offset[0] = -1; f_offset[0] <= 1; f_offset[0]++) {
                    float weight_prod = 1.0f;
                    long long c_final_idx[3];
                    for (i_dim = 0; i_dim < 3; i_dim++) {
                        c_final_idx[i_dim] = Mod(mid_cell[i_dim] + f_offset[i_dim], n);
                        weight_prod *= w_basis[i_dim][f_offset[i_dim] + 1];
                    }
                    long long N_sq = static_cast<long long>(n) * n;
                    long long flat_idx = c_final_idx[0] + c_final_idx[1] * n + c_final_idx[2] * N_sq;
                    if (flat_idx < 0 || flat_idx >= N_sq * n) continue;
                    result += weight_prod * noiseTileData[flat_idx];
                }
            }
        }
        return result;
    }
};

float* WaveletNoise::noiseTileData = nullptr;
int WaveletNoise::noiseTileSize = 0;

void generateSingleBandDiagnostic(int image_output_size = 128) {
    std::cout << "\n=== GENERATING DIAGNOSTIC IMAGES ===" << std::endl;
    std::cout << "Output size: " << image_output_size << "x" << image_output_size << std::endl;
    
    if (WaveletNoise::noiseTileSize == 0 || WaveletNoise::noiseTileData == nullptr) {
        std::cerr << "Error: Noise tile not initialized." << std::endl;
        return;
    }

    std::vector<std::vector<float>> baseBandImage(image_output_size, std::vector<float>(image_output_size));
    int tile_dim = WaveletNoise::noiseTileSize;

    // Generate noise samples
    float noise_coord_scale = static_cast<float>(tile_dim) / image_output_size;
    
    std::cout << "Sampling noise with scale factor: " << noise_coord_scale << std::endl;

    for (int y = 0; y < image_output_size; y++) {
        for (int x = 0; x < image_output_size; x++) {
            float p_noise[3] = {
                x * noise_coord_scale,
                y * noise_coord_scale,
                0.5f * tile_dim
            };
            baseBandImage[y][x] = WaveletNoise::WNoise(p_noise);
        }
    }
    
    // Calculate statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0, sum_sq = 0.0;
    
    for (int y = 0; y < image_output_size; y++) {
        for (int x = 0; x < image_output_size; x++) {
            float val = baseBandImage[y][x];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
            sum_sq += val * val;
        }
    }
    
    int total_pixels = image_output_size * image_output_size;
    float avg = sum / total_pixels;
    float var = (sum_sq / total_pixels) - avg * avg;
    
    std::cout << "\nDiagnostic image statistics:" << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
    std::cout << "  Average: " << avg << ", Variance: " << var << std::endl;
    std::cout << "  Std Dev: " << sqrt(var) << std::endl;
    
    saveBMP("diagnostic_base_band.bmp", baseBandImage);
    
    // FFT analysis
    if ((image_output_size & (image_output_size-1)) == 0) {
        std::vector<std::vector<Complex>> baseSpectrum;
        fft2D(baseBandImage, baseSpectrum);
        saveSpectrum("diagnostic_base_band_spectrum.bmp", baseSpectrum);
        analyzeFrequencyContent(baseSpectrum, "diagnostic_base_band");
    }
    
    std::cout << "\nDiagnostic images saved." << std::endl;
}

// ===== Main program =====
int main() {
    std::cout << "=== ENHANCED WAVELET NOISE DIAGNOSTIC ===" << std::endl;
    std::cout << "This version includes detailed frequency analysis" << std::endl;
    
    int tile_generation_size = 128;
    std::cout << "\nGenerating noise tile (" << tile_generation_size << "^3)..." << std::endl;
    
    WaveletNoise::GenerateNoiseTile(tile_generation_size);
    
    if (WaveletNoise::noiseTileData == nullptr || WaveletNoise::noiseTileSize == 0) {
        std::cerr << "Critical Error: Noise tile generation failed." << std::endl;
        return 1;
    }
    
    int diagnostic_image_output_size = 256;
    generateSingleBandDiagnostic(diagnostic_image_output_size);
    
    std::cout << "\n=== DIAGNOSTICS COMPLETE ===" << std::endl;
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "Intermediate stages (z=" << tile_generation_size/2 << " slices):" << std::endl;
    std::cout << "  - debug_R_initial_slice_z*.bmp/csv/spectrum.bmp" << std::endl;
    std::cout << "  - debug_R_down_up_slice_z*.bmp/csv/spectrum.bmp" << std::endl;
    std::cout << "  - debug_N_intermediate_slice_z*.bmp/csv/spectrum.bmp" << std::endl;
    std::cout << "  - debug_N_final_slice_z*.bmp/csv/spectrum.bmp" << std::endl;
    std::cout << "\nFinal evaluation:" << std::endl;
    std::cout << "  - diagnostic_base_band.bmp/csv/spectrum.bmp" << std::endl;
    
    // Cleanup
    if (WaveletNoise::noiseTileData) {
        free(WaveletNoise::noiseTileData);
        WaveletNoise::noiseTileData = nullptr;
    }

    return 0;
}