// wavelet_noise.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <complex>
#include <cstring> // For memcpy
#include <numeric> // For std::iota, std::accumulate
#include <limits>  // For std::numeric_limits

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    for (int y = height - 1; y >= 0; y--) { // BMPs are bottom-up
        for (int x = 0; x < width; x++) {
            float val = data[y][x];
            if (std::isnan(val) || std::isinf(val)) val = 0.0f;
            // Normalize from [-1, 1] to [0, 255]
            uint8_t value = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (val + 1.0f) * 127.5f)));
            row_buffer[x * 3 + 0] = value; // B
            row_buffer[x * 3 + 1] = value; // G
            row_buffer[x * 3 + 2] = value; // R
        }
        for (int i = width * 3; i < row_stride; ++i) {
            row_buffer[i] = 0; // Padding
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

// ===== 數學工具函數 =====
inline int Mod(int x, int n) {
    if (n == 0) { std::cerr << "Error: Modulo by zero (n=0)." << std::endl; return 0; }
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

// ===== FFT 實現 =====
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
        std::cerr << "FFT Error: Image dimensions (" << width << "x" << height << ") must be non-zero and powers of 2 for this FFT implementation." << std::endl;
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
        std::cerr << "Error: Empty spectrum data for saveSpectrum: " << filename_bmp << std::endl;
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

    std::vector<std::vector<float>> log_magnitudes_for_bmp(height, std::vector<float>(width));
    float min_log_m = std::numeric_limits<float>::max();
    float max_log_m = std::numeric_limits<float>::lowest();
    bool first_finite_log_m = true;
    const float FACTOR = 10000.0f; // Increased factor for more contrast

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag_val = shifted_magnitudes[i][j]; // Use already shifted magnitudes
            float current_log_mag;
            if (maxMag > 1e-9f) { // Avoid division by zero if maxMag is tiny
                 current_log_mag = std::log(1.0f + mag_val / maxMag * FACTOR);
            } else {
                current_log_mag = 0.0f; // log(1) is 0
            }
            log_magnitudes_for_bmp[i][j] = current_log_mag;

            if (std::isfinite(current_log_mag)) {
                if (first_finite_log_m || current_log_mag < min_log_m) min_log_m = current_log_mag;
                if (first_finite_log_m || current_log_mag > max_log_m) max_log_m = current_log_mag;
                first_finite_log_m = false;
            }
        }
    }
    
    if (first_finite_log_m) { // All log_magnitudes were NaN/Inf or empty
        min_log_m = 0.0f;
        max_log_m = 1.0f; // Arbitrary small positive range
    }
    if (std::abs(max_log_m - min_log_m) < 1e-6f) { // If range is still too small, expand it
        max_log_m = min_log_m + 1.0f; 
    }

    std::vector<std::vector<float>> output_img(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (std::isfinite(log_magnitudes_for_bmp[i][j])) {
                if (max_log_m - min_log_m > 1e-6f) {
                    output_img[i][j] = 2.0f * (log_magnitudes_for_bmp[i][j] - min_log_m) / (max_log_m - min_log_m) - 1.0f;
                } else { 
                    output_img[i][j] = -1.0f; // Black if range is too small
                }
            } else {
                output_img[i][j] = -1.0f; // Black for NaN/Inf
            }
        }
    }
    saveBMP(filename_bmp, output_img);
}


// ===== 協助函數：計算統計數據 =====
struct DataStats {
    float avg = 0.0f; float var = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    long long count_nan_inf = 0;
};

DataStats calculateStats(const float* data, long long total_size, const std::string& name) {
    DataStats stats;
    if (total_size == 0) { std::cerr << "Warning: Stats for zero-sized data: " << name << std::endl; return stats; }
    double sum = 0.0, sum_sq = 0.0;
    long long valid_count = 0;
    for (long long i = 0; i < total_size; ++i) {
        float val = data[i];
        if (std::isnan(val) || std::isinf(val)) { stats.count_nan_inf++; continue; }
        valid_count++; sum += val; sum_sq += static_cast<double>(val) * val;
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
    }
    if (valid_count > 0) {
        stats.avg = static_cast<float>(sum / valid_count);
        stats.var = static_cast<float>((sum_sq / valid_count) - static_cast<double>(stats.avg) * stats.avg);
    } else { std::cerr << "Warning: No valid data for stats: " << name << std::endl; stats.min_val=0; stats.max_val=0;}
    std::cout << name << " stats: avg=" << stats.avg << " var=" << stats.var
              << " stddev=" << (stats.var > 0 ? sqrt(stats.var) : 0)
              << " min=" << stats.min_val << " max=" << stats.max_val;
    if (stats.count_nan_inf > 0) std::cout << " (NaN/Inf: " << stats.count_nan_inf << ")";
    std::cout << std::endl;
    return stats;
}

void saveTileSlice(const std::string& filename_prefix, int slice_z, const float* tile_data, int n) {
    if (!tile_data || n <= 0 || slice_z < 0 || slice_z >= n) {
        std::cerr << "Error: Invalid args for saveTileSlice: " << filename_prefix << std::endl; return;
    }
    std::vector<std::vector<float>> slice_image(n, std::vector<float>(n));
    float current_min_val = std::numeric_limits<float>::max();
    float current_max_val = std::numeric_limits<float>::lowest();
    bool all_same = true; float first_val = 0.0f;

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            long long index = static_cast<long long>(x) + static_cast<long long>(y) * n + static_cast<long long>(slice_z) * n * n;
            if (index >= static_cast<long long>(n)*n*n) { std::cerr << "Error: Index out of bounds in saveTileSlice." << std::endl; slice_image[y][x] = 0.0f; continue; }
            float val = tile_data[index]; slice_image[y][x] = val;
            if (std::isnan(val) || std::isinf(val)) continue;
            if (y == 0 && x == 0) first_val = val; else if (std::abs(val - first_val) > 1e-6f) all_same = false;
            if (val < current_min_val) current_min_val = val; if (val > current_max_val) current_max_val = val;
        }
    }
    if (!all_same && (current_max_val - current_min_val > 1e-6f) && std::isfinite(current_min_val) && std::isfinite(current_max_val) ) {
        for(int y=0; y<n; ++y) for(int x=0; x<n; ++x) {
            slice_image[y][x] = std::isfinite(slice_image[y][x]) ? 2.0f * (slice_image[y][x] - current_min_val) / (current_max_val - current_min_val) - 1.0f : 0.0f;
        }
    } else { for(int y=0; y<n; ++y) for(int x=0; x<n; ++x) slice_image[y][x] = 0.0f; }

    saveBMP(filename_prefix + "_slice_z" + std::to_string(slice_z) + ".bmp", slice_image);
    
    if (n > 0 && (n & (n - 1)) == 0) { // FFT for power-of-2 sizes
        std::vector<std::vector<Complex>> spectrum_data;
        fft2D(slice_image, spectrum_data); // Use normalized slice_image for FFT
        saveSpectrum(filename_prefix + "_slice_z" + std::to_string(slice_z) + "_spectrum.bmp", spectrum_data);
    } else if (n > 0) { std::cout << "Skipping spectrum for " << filename_prefix << " because size " << n << " is not power of 2." << std::endl;}
}

// ===== Wavelet Noise 實現 =====
class WaveletNoise {
public:
    static float* noiseTileData;
    static int noiseTileSize;
    static constexpr int ARAD = 16;

    static void GenerateNoiseTile(int n_target, int /*olap*/ = 0) {
        if (n_target <= 0) { std::cerr << "Error: GenerateNoiseTile n=" << n_target << " invalid." << std::endl; noiseTileData = nullptr; noiseTileSize = 0; return; }
        int n = n_target; // Use 'n' internally for tile dimension
        if (n % 2) n++;
        noiseTileSize = n;
        long long num_elements = static_cast<long long>(n) * n * n;
        size_t sz_bytes = num_elements * sizeof(float);

        float* r_initial  = (float*)malloc(sz_bytes);
        float* temp1      = (float*)malloc(sz_bytes);
        float* r_down_up  = (float*)malloc(sz_bytes); // Was temp2, renamed for clarity
        float* n_final    = (float*)malloc(sz_bytes); // Was noise_buffer

        if (!r_initial || !temp1 || !r_down_up || !n_final) {
            std::cerr << "Error: Malloc failed in GenerateNoiseTile." << std::endl;
            free(r_initial); free(temp1); free(r_down_up); free(n_final);
            noiseTileData = nullptr; noiseTileSize = 0; return;
        }

        std::cout << "Step 1: Filling R_initial with Gaussian noise..." << std::endl;
        for (long long i = 0; i < num_elements; i++) r_initial[i] = gaussianNoise();
        calculateStats(r_initial, num_elements, "R_initial");
        if ( (n&(n-1))==0 ) saveTileSlice("debug_R_initial", n / 2, r_initial, n);


        std::cout << "Steps 2 & 3: Creating R_down_up (Downsample & Upsample R_initial)..." << std::endl;
        // Pass X (R_initial -> temp1 (down) -> r_down_up (up))
        for (int iy = 0; iy < n; iy++) for (int iz = 0; iz < n; iz++) {
            long long base_idx = static_cast<long long>(iy) * n + static_cast<long long>(iz) * n * n;
            Downsample(&r_initial[base_idx], &temp1[base_idx], n, 1);
            Upsample(&temp1[base_idx], &r_down_up[base_idx], n, 1);
        }
        // Pass Y (r_down_up -> temp1 (down) -> r_down_up (up, overwrite))
        for (int ix = 0; ix < n; ix++) for (int iz = 0; iz < n; iz++) {
            long long base_idx = static_cast<long long>(ix) + static_cast<long long>(iz) * n * n;
            Downsample(&r_down_up[base_idx], &temp1[base_idx], n, n);
            Upsample(&temp1[base_idx], &r_down_up[base_idx], n, n);
        }
        // Pass Z (r_down_up -> temp1 (down) -> r_down_up (up, overwrite))
        for (int ix = 0; ix < n; ix++) for (int iy = 0; iy < n; iy++) {
            long long base_idx = static_cast<long long>(ix) + static_cast<long long>(iy) * n;
            Downsample(&r_down_up[base_idx], &temp1[base_idx], n, n * n);
            Upsample(&temp1[base_idx], &r_down_up[base_idx], n, n * n);
        }
        calculateStats(r_down_up, num_elements, "R_down_up");
        if ( (n&(n-1))==0 ) saveTileSlice("debug_R_down_up", n / 2, r_down_up, n);
        
        std::cout << "Step 4: N_intermediate = R_initial - R_down_up..." << std::endl;
        for (long long i = 0; i < num_elements; i++) n_final[i] = r_initial[i] - r_down_up[i];
        calculateStats(n_final, num_elements, "N_intermediate (before offset sum)");
        if ( (n&(n-1))==0 ) saveTileSlice("debug_N_intermediate", n / 2, n_final, n);

        std::cout << "Step 5: Avoiding even/odd variance (N_final = N_intermediate + N_offset)..." << std::endl;
        int offset_val = n / 2; if (offset_val % 2 == 0) offset_val++;
        // Use temp1 for N_offset storage
        for (int ix = 0; ix < n; ix++) for (int iy = 0; iy < n; iy++) for (int iz = 0; iz < n; iz++) {
            long long target_idx = static_cast<long long>(ix) + static_cast<long long>(iy) * n + static_cast<long long>(iz) * n * n;
            long long src_idx = static_cast<long long>(Mod(ix + offset_val, n)) +
                                static_cast<long long>(Mod(iy + offset_val, n)) * n +
                                static_cast<long long>(Mod(iz + offset_val, n)) * n * n;
            if (src_idx < 0 || src_idx >= num_elements) { std::cerr << "Error: N_offset src_idx out of bounds." << std::endl; temp1[target_idx] = 0.0f; }
            else { temp1[target_idx] = n_final[src_idx]; } // N_final currently holds N_intermediate
        }
        for (long long i = 0; i < num_elements; i++) n_final[i] += temp1[i];
        calculateStats(n_final, num_elements, "N_final (after offset sum)");
        if ( (n&(n-1))==0 ) saveTileSlice("debug_N_final", n / 2, n_final, n);
        
        free(r_initial); free(temp1); free(r_down_up);
        noiseTileData = n_final; // Ownership transferred
        std::cout << "Noise tile generation complete." << std::endl;
    }

    static void Downsample(const float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) { std::cerr << "Error: Invalid args to Downsample." << std::endl; return; }
        float aCoeffs[2*ARAD] = {
            0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
            -0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
            0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
            0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
        };
        const float* a_centered = &aCoeffs[ARAD]; // Pointer to the center of the symmetric filter
        for (int i_out = 0; i_out < n / 2; i_out++) {
            long long to_idx = static_cast<long long>(i_out) * stride;
            to[to_idx] = 0.0f;
            // Paper: to[i] = sum_{k} a[k-2i] * from[k]
            // For our centered a_centered: sum_{k_filt_tap} a_centered[k_filt_tap] * from[2*i_out + k_filt_tap]
            for (int k_filter_tap_idx = -ARAD; k_filter_tap_idx < ARAD; ++k_filter_tap_idx) { // Iterate over filter taps
                int k_input_original_idx = 2 * i_out + k_filter_tap_idx; // Corresponding 'from' index before Mod
                long long from_idx_in_dimension = Mod(k_input_original_idx, n);
                long long from_element_absolute_idx = from_idx_in_dimension * stride;
                to[to_idx] += a_centered[k_filter_tap_idx] * from[from_element_absolute_idx];
            }
        }
    }

    static void Upsample(const float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) { std::cerr << "Error: Invalid args to Upsample." << std::endl; return; }
        float pCoeffs[4] = { 0.25f, 0.75f, 0.75f, 0.25f };
        const float* p_centered = &pCoeffs[2]; // Center for p[i-2k] style indexing
        for (int i_out = 0; i_out < n; i_out++) { // For each output 'to[i_out]'
            long long to_idx = static_cast<long long>(i_out) * stride;
            to[to_idx] = 0.0f;
            // Paper: to[i] = sum_{k} p[i-2k] * from[k]
            // Loop over k (input indices for 'from', which is half-size)
            // The filter p is short, only pCoeffs[0..3] are non-zero.
            // p_centered[idx] means pCoeffs[2+idx]. For valid access, 2+idx in [0,3] -> idx in [-2,1]
            // So, i_out - 2*k_in must be in [-2,1].
            for (int k_in_offset = 0; k_in_offset <= 1; ++k_in_offset) { // k_in will take two values for each i_out
                 int k_in = i_out/2 - 1 + k_in_offset; // Adjusted to make p_filter_idx range correctly from -1 to 2
                                                      // when centered access p_centered[p_filter_idx]
                                                      // (Original paper has p as {p_0,p_1,p_2,p_3}, not centered access)
                                                      // Let's stick to original paper's p indexing for filter coefficients:
                                                      // p[0]=0.25, p[1]=0.75, p[2]=0.75, p[3]=0.25
                                                      // For to[i_out], we sum p[i_out-2*k_in]*from[k_in]
                                                      // The relevant k_in values are those for which i_out-2*k_in is in [0,3]
                                                      // E.g., i_out=0 -> -2*k_in in [0,3] -> k_in = 0 (p[0]*f[0]), k_in=-1 (p[2]*f[-1])
                                                      // This is getting complicated. Let's use the provided code's loop:
                // This loop for k corresponds to the Appendix: for (int k=i/2; k<=i/2+1; k++)
                // which correctly picks two 'from' samples for each 'to' sample.
                int k_from_idx_local_start = i_out / 2; // equivalent to floor(i_out/2)
                for (int k_offset_idx = 0; k_offset_idx <=1; ++k_offset_idx ) {
                    int k_from_local = k_from_idx_local_start - k_offset_idx; // this makes k range over floor(i/2) and floor(i/2)-1 effectively

                    int p_coeff_lookup_idx = i_out - 2 * k_from_local;
                    // We need p_coeff_lookup_idx to be in range of pCoeffs {0,1,2,3}
                    // The p_centered access: p_centered[p_idx] == pCoeffs[2+p_idx].
                    // We want p_idx to align with the paper's direct coefficients, so use pCoeffs directly.
                    if (p_coeff_lookup_idx >= 0 && p_coeff_lookup_idx < 4) {
                         long long from_idx_mod = Mod(k_from_local, n/2);
                         long long from_element_absolute_idx = from_idx_mod * stride;
                         to[to_idx] += pCoeffs[p_coeff_lookup_idx] * from[from_element_absolute_idx];
                    }
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
            t_param = mid_cell[i_dim] - (p[i_dim] - 0.5f); // t parameter for quadratic B-spline
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
                        weight_prod *= w_basis[i_dim][f_offset[i_dim] + 1]; // f_offset is -1,0,1; w_basis index is 0,1,2
                    }
                    long long N_sq = static_cast<long long>(n) * n;
                    long long flat_idx = c_final_idx[0] + c_final_idx[1] * n + c_final_idx[2] * N_sq;
                    if (flat_idx < 0 || flat_idx >= N_sq * n) { /* Error */ continue; }
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
    std::cout << "\nGenerating single band diagnostic images (output size " << image_output_size << "x" << image_output_size << ")..." << std::endl;
    if (WaveletNoise::noiseTileSize == 0 || WaveletNoise::noiseTileData == nullptr) {
        std::cerr << "Error: Noise tile not initialized for single band diagnostic." << std::endl;
        return;
    }
    if ((image_output_size & (image_output_size-1)) != 0) {
         std::cerr << "Warning: diagnostic image_output_size " << image_output_size << " is not a power of 2. FFT might fail or be inaccurate." << std::endl;
    }

    std::vector<std::vector<float>> baseBandImage(image_output_size, std::vector<float>(image_output_size));
    int tile_dim = WaveletNoise::noiseTileSize; // Actual dimension of the generated noise tile data

    // Generate N(x) where x spans roughly one period of the noise tile in image space.
    // Map image coords [0, image_output_size-1] to noise coords [0, tile_dim-1]
    float noise_coord_scale = static_cast<float>(tile_dim) / image_output_size;

    for (int y = 0; y < image_output_size; y++) {
        for (int x = 0; x < image_output_size; x++) {
            float p_noise[3] = {
                x * noise_coord_scale,
                y * noise_coord_scale,
                0.5f * tile_dim // Pick a z-slice, e.g., middle of the tile
            };
            baseBandImage[y][x] = WaveletNoise::WNoise(p_noise);
        }
    }
    saveBMP("diagnostic_base_band.bmp", baseBandImage);
    std::vector<std::vector<Complex>> baseSpectrum;
    fft2D(baseBandImage, baseSpectrum); // FFT of the normalized image data
    saveSpectrum("diagnostic_base_band_spectrum.bmp", baseSpectrum);
    std::cout << "Diagnostic base band and spectrum saved." << std::endl;
}


// ===== 主程式 =====
int main() {
    std::cout << "=== Wavelet Noise Diagnostic Mode ===" << std::endl;
    
    int tile_generation_size = 128; // Must be power of 2 for FFT in saveTileSlice
    std::cout << "Generating noise tile (" << tile_generation_size << "x" 
              << tile_generation_size << "x" << tile_generation_size << ")..." << std::endl;
    WaveletNoise::GenerateNoiseTile(tile_generation_size); 
    
    if (WaveletNoise::noiseTileData == nullptr || WaveletNoise::noiseTileSize == 0) {
        std::cerr << "Critical Error: Noise tile not generated. Aborting." << std::endl;
        return 1;
    }
    
    int diagnostic_image_output_size = 256; // Can be different from tile_generation_size
                                          // but also should be power of 2 for its own FFT
    generateSingleBandDiagnostic(diagnostic_image_output_size);
    
    std::cout << "\n--- Wavelet Noise Diagnostics Finished ---" << std::endl;
    std::cout << "Debug files generated (slices from tile generation process):" << std::endl;
    std::cout << "  debug_R_initial_slice_z" << tile_generation_size/2 << ".bmp/.csv/_spectrum.bmp" << std::endl;
    std::cout << "  debug_R_down_up_slice_z" << tile_generation_size/2 << ".bmp/.csv/_spectrum.bmp" << std::endl;
    std::cout << "  debug_N_intermediate_slice_z" << tile_generation_size/2 << ".bmp/.csv/_spectrum.bmp" << std::endl;
    std::cout << "  debug_N_final_slice_z" << tile_generation_size/2 << ".bmp/.csv/_spectrum.bmp" << std::endl;
    std::cout << "Diagnostic files generated (evaluation of final N_final tile):" << std::endl;
    std::cout << "  diagnostic_base_band.bmp/.csv/_spectrum.bmp" << std::endl;

    // Cleanup
    if (WaveletNoise::noiseTileData) {
        free(WaveletNoise::noiseTileData);
        WaveletNoise::noiseTileData = nullptr;
    }

    return 0;
}