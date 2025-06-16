// wavelet_noise_fixed.cpp
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
    if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0) {
        std::cerr << "FFT Error: Dimensions must be powers of 2." << std::endl;
        return;
    }
    output.assign(height, std::vector<Complex>(width));
    for(int i=0; i<height; ++i) {
        for(int j=0; j<width; ++j) {
            output[i][j] = input[i][j];
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
    if (spectrum.empty() || spectrum[0].empty()) return;
    
    int height = spectrum.size();
    int width = spectrum[0].size();

    std::vector<std::vector<float>> magnitudes(height, std::vector<float>(width));
    float maxMag = 0.0f;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = std::abs(spectrum[i][j]);
            magnitudes[i][j] = mag;
            if (mag > maxMag) maxMag = mag;
        }
    }

    // Shift spectrum
    std::vector<std::vector<float>> shifted_magnitudes(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            shifted_magnitudes[i][j] = magnitudes[(i + height / 2) % height][(j + width / 2) % width];
        }
    }
    
    std::string filename_csv = filename_bmp;
    size_t dot_pos = filename_csv.rfind(".bmp");
    if (dot_pos != std::string::npos) filename_csv.replace(dot_pos, 4, "_magnitude.csv");
    else filename_csv += "_magnitude.csv";
    saveDataAsCSV(filename_csv, shifted_magnitudes);

    // Log scale for visualization
    std::vector<std::vector<float>> output_img(height, std::vector<float>(width));
    const float FACTOR = 10000.0f;
    float min_log = std::numeric_limits<float>::max();
    float max_log = std::numeric_limits<float>::lowest();

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float log_mag = std::log(1.0f + shifted_magnitudes[i][j] / maxMag * FACTOR);
            output_img[i][j] = log_mag;
            if (log_mag < min_log) min_log = log_mag;
            if (log_mag > max_log) max_log = log_mag;
        }
    }
    
    if (max_log - min_log > 1e-6f) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output_img[i][j] = 2.0f * (output_img[i][j] - min_log) / (max_log - min_log) - 1.0f;
            }
        }
    }
    
    saveBMP(filename_bmp, output_img);
}

struct DataStats {
    float avg = 0.0f;
    float var = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float energy = 0.0f;
};

DataStats calculateStats(const float* data, long long total_size, const std::string& name) {
    DataStats stats;
    if (total_size == 0) return stats;
    
    double sum = 0.0, sum_sq = 0.0;
    for (long long i = 0; i < total_size; ++i) {
        float val = data[i];
        if (!std::isfinite(val)) continue;
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
    }
    
    stats.avg = static_cast<float>(sum / total_size);
    stats.var = static_cast<float>((sum_sq / total_size) - static_cast<double>(stats.avg) * stats.avg);
    stats.energy = static_cast<float>(sum_sq);
    
    std::cout << name << " stats: avg=" << stats.avg << " var=" << stats.var
              << " stddev=" << sqrt(stats.var) << " min=" << stats.min_val 
              << " max=" << stats.max_val << " energy=" << stats.energy << std::endl;
    
    return stats;
}

void analyzeFrequencyContent(const std::vector<std::vector<Complex>>& spectrum, const std::string& name) {
    if (spectrum.empty() || spectrum[0].empty()) return;
    
    int height = spectrum.size();
    int width = spectrum[0].size();
    int cy = height / 2;
    int cx = width / 2;
    
    float dc_energy = 0.0f;
    float low_freq_energy = 0.0f;
    float mid_freq_energy = 0.0f;
    float high_freq_energy = 0.0f;
    float total_energy = 0.0f;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = std::abs(spectrum[i][j]);
            float energy = mag * mag;
            total_energy += energy;
            
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
}

void saveTileSlice(const std::string& filename_prefix, int slice_z, const float* tile_data, int n) {
    if (!tile_data || n <= 0 || slice_z < 0 || slice_z >= n) return;
    
    std::vector<std::vector<float>> slice_image(n, std::vector<float>(n));
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            long long index = x + y * n + slice_z * n * n;
            float val = tile_data[index];
            slice_image[y][x] = val;
            if (std::isfinite(val)) {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
    }
    
    if (max_val - min_val > 1e-6f) {
        for(int y=0; y<n; ++y) {
            for(int x=0; x<n; ++x) {
                slice_image[y][x] = 2.0f * (slice_image[y][x] - min_val) / (max_val - min_val) - 1.0f;
            }
        }
    }

    saveBMP(filename_prefix + "_slice_z" + std::to_string(slice_z) + ".bmp", slice_image);
    
    if (n > 0 && (n & (n - 1)) == 0) {
        std::vector<std::vector<Complex>> spectrum_data;
        fft2D(slice_image, spectrum_data);
        saveSpectrum(filename_prefix + "_slice_z" + std::to_string(slice_z) + "_spectrum.bmp", spectrum_data);
        analyzeFrequencyContent(spectrum_data, filename_prefix);
    }
}

// ===== Wavelet Noise Implementation =====
class WaveletNoise {
public:
    static float* noiseTileData;
    static int noiseTileSize;
    static constexpr int ARAD = 16;

    static void GenerateNoiseTile(int n_target, int /*olap*/ = 0) {
        if (n_target <= 0) return;
        
        int n = n_target;
        if (n % 2) n++;
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
            return;
        }

        std::cout << "\n=== WAVELET NOISE GENERATION (FIXED VERSION) ===" << std::endl;
        std::cout << "Tile size: " << n << "x" << n << "x" << n << std::endl;

        // Step 1: Fill with Gaussian noise
        std::cout << "\nStep 1: Filling with Gaussian noise..." << std::endl;
        for (long long i = 0; i < num_elements; i++) {
            r_initial[i] = gaussianNoise();
        }
        DataStats stats_r_initial = calculateStats(r_initial, num_elements, "R_initial");
        saveTileSlice("fixed_R_initial", n / 2, r_initial, n);

        // Steps 2 & 3: Downsample and Upsample with CORRECTED normalization
        std::cout << "\nSteps 2 & 3: Downsampling and Upsampling (with correct normalization)..." << std::endl;
        
        memcpy(r_down_up, r_initial, sz_bytes);
        
        // Process each dimension
        for (int iy = 0; iy < n; iy++) {
            for (int iz = 0; iz < n; iz++) {
                long long base_idx = iy * n + iz * n * n;
                Downsample(&r_down_up[base_idx], &temp1[base_idx], n, 1);
                Upsample(&temp1[base_idx], &r_down_up[base_idx], n, 1);
            }
        }
        
        for (int ix = 0; ix < n; ix++) {
            for (int iz = 0; iz < n; iz++) {
                long long base_idx = ix + iz * n * n;
                Downsample(&r_down_up[base_idx], &temp1[base_idx], n, n);
                Upsample(&temp1[base_idx], &r_down_up[base_idx], n, n);
            }
        }
        
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                long long base_idx = ix + iy * n;
                Downsample(&r_down_up[base_idx], &temp1[base_idx], n, n * n);
                Upsample(&temp1[base_idx], &r_down_up[base_idx], n, n * n);
            }
        }
        
        DataStats stats_r_down_up = calculateStats(r_down_up, num_elements, "R_down_up");
        saveTileSlice("fixed_R_down_up", n / 2, r_down_up, n);
        
        // Step 4: Subtract
        std::cout << "\nStep 4: Computing N_intermediate = R_initial - R_down_up..." << std::endl;
        for (long long i = 0; i < num_elements; i++) {
            n_final[i] = r_initial[i] - r_down_up[i];
        }
        DataStats stats_n_intermediate = calculateStats(n_final, num_elements, "N_intermediate");
        saveTileSlice("fixed_N_intermediate", n / 2, n_final, n);

        // Step 5: Add offset
        std::cout << "\nStep 5: Adding offset for variance correction..." << std::endl;
        int offset_val = n / 2;
        if (offset_val % 2 == 0) offset_val++;
        
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                for (int iz = 0; iz < n; iz++) {
                    long long idx = ix + iy * n + iz * n * n;
                    long long offset_idx = Mod(ix + offset_val, n) + 
                                         Mod(iy + offset_val, n) * n + 
                                         Mod(iz + offset_val, n) * n * n;
                    temp1[idx] = n_final[offset_idx];
                }
            }
        }
        
        for (long long i = 0; i < num_elements; i++) {
            n_final[i] += temp1[i];
        }
        
        DataStats stats_n_final = calculateStats(n_final, num_elements, "N_final");
        saveTileSlice("fixed_N_final", n / 2, n_final, n);
        
        std::cout << "\n=== ENERGY ANALYSIS ===" << std::endl;
        std::cout << "Energy ratio (R_down_up / R_initial): " << stats_r_down_up.energy / stats_r_initial.energy << std::endl;
        std::cout << "Energy ratio (N_final / R_initial): " << stats_n_final.energy / stats_r_initial.energy << std::endl;
        std::cout << "This should show R_down_up has ~80-90% of original energy." << std::endl;
        
        free(r_initial);
        free(temp1);
        free(r_down_up);
        noiseTileData = n_final;
    }

    // FIXED Downsample function with proper normalization
    static void Downsample(const float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) return;
        
        // Original coefficients
        float aCoeffs[2*ARAD] = {
            0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
            -0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
            0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
            0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
        };
        
        // CRITICAL FIX: Normalize coefficients so they sum to sqrt(2)
        const float SQRT2 = std::sqrt(2.0f);
        for (int i = 0; i < 2*ARAD; i++) {
            aCoeffs[i] *= SQRT2;  // Multiply by sqrt(2) since original sum is ~1.0
        }
        
        const float* a = &aCoeffs[ARAD];
        
        for (int i = 0; i < n / 2; i++) {
            to[i * stride] = 0.0f;
            for (int k = -ARAD; k < ARAD; k++) {
                int idx = Mod(2*i + k, n);
                to[i * stride] += a[k] * from[idx * stride];
            }
        }
    }

    static void Upsample(const float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) return;
        
        // Synthesis coefficients (already correct, sum to 2.0)
        float pCoeffs[4] = { 0.25f, 0.75f, 0.75f, 0.25f };
        const float* p = &pCoeffs[2];
        
        for (int i = 0; i < n; i++) {
            to[i * stride] = 0.0f;
            for (int k = i/2; k <= i/2 + 1; k++) {
                int p_idx = i - 2*k;
                if (p_idx >= -2 && p_idx <= 1) {
                    to[i * stride] += p[p_idx] * from[Mod(k, n/2) * stride];
                }
            }
        }
    }

    static float WNoise(float p[3]) {
        int n = noiseTileSize;
        if (n == 0 || !noiseTileData) return 0.0f;
        
        int f[3], c[3], mid[3];
        float w[3][3], t, result = 0.0f;

        for (int i = 0; i < 3; i++) {
            mid[i] = static_cast<int>(std::ceil(p[i] - 0.5f));
            t = mid[i] - (p[i] - 0.5f);
            w[i][0] = t * t / 2.0f;
            w[i][2] = (1.0f - t) * (1.0f - t) / 2.0f;
            w[i][1] = 1.0f - w[i][0] - w[i][2];
        }

        for (f[2] = -1; f[2] <= 1; f[2]++) {
            for (f[1] = -1; f[1] <= 1; f[1]++) {
                for (f[0] = -1; f[0] <= 1; f[0]++) {
                    float weight = 1.0f;
                    for (int i = 0; i < 3; i++) {
                        c[i] = Mod(mid[i] + f[i], n);
                        weight *= w[i][f[i] + 1];
                    }
                    result += weight * noiseTileData[c[0] + c[1] * n + c[2] * n * n];
                }
            }
        }
        return result;
    }
};

float* WaveletNoise::noiseTileData = nullptr;
int WaveletNoise::noiseTileSize = 0;

void generateSingleBandDiagnostic(int image_size = 128) {
    std::cout << "\n=== GENERATING DIAGNOSTIC IMAGE ===" << std::endl;
    
    if (WaveletNoise::noiseTileSize == 0 || WaveletNoise::noiseTileData == nullptr) {
        std::cerr << "Error: Noise tile not initialized." << std::endl;
        return;
    }

    std::vector<std::vector<float>> image(image_size, std::vector<float>(image_size));
    int n = WaveletNoise::noiseTileSize;
    float scale = static_cast<float>(n) / image_size;

    for (int y = 0; y < image_size; y++) {
        for (int x = 0; x < image_size; x++) {
            float p[3] = { x * scale, y * scale, 0.5f * n };
            image[y][x] = WaveletNoise::WNoise(p);
        }
    }
    
    saveBMP("fixed_diagnostic_base_band.bmp", image);
    
    if ((image_size & (image_size-1)) == 0) {
        std::vector<std::vector<Complex>> spectrum;
        fft2D(image, spectrum);
        saveSpectrum("fixed_diagnostic_base_band_spectrum.bmp", spectrum);
        analyzeFrequencyContent(spectrum, "fixed_diagnostic_base_band");
    }
}

int main() {
    std::cout << "=== FIXED WAVELET NOISE TEST ===" << std::endl;
    std::cout << "This version includes proper filter normalization." << std::endl;
    
    WaveletNoise::GenerateNoiseTile(128);
    
    if (WaveletNoise::noiseTileData) {
        generateSingleBandDiagnostic(256);
        free(WaveletNoise::noiseTileData);
        WaveletNoise::noiseTileData = nullptr;
    }
    
    return 0;
}