#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <complex>
#include <cstring>
#include <numeric> // For std::accumulate
#include <limits>  // For std::numeric_limits

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
    // Calculate image_size carefully to avoid overflow with large dimensions
    // Each pixel is 3 bytes. Row size needs padding to multiple of 4.
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
            // Clamp data[y][x] to avoid issues if it's NaN or Inf before normalization
            float val = data[y][x];
            if (std::isnan(val) || std::isinf(val)) {
                // std::cerr << "Warning: NaN or Inf found in data for saveBMP at (" << x << "," << y << ") in " << filename << ". Setting to 0." << std::endl;
                val = 0.0f; // Replace NaN/Inf with a neutral value
            }
            uint8_t value = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, (val + 1.0f) * 127.5f)));
            row_buffer[x * 3 + 0] = value; // B
            row_buffer[x * 3 + 1] = value; // G
            row_buffer[x * 3 + 2] = value; // R
        }
        // Padding already handled by row_stride, but ensure buffer is zeroed if needed
        for (int i = width * 3; i < row_stride; ++i) {
            row_buffer[i] = 0;
        }
        file.write(reinterpret_cast<char*>(row_buffer.data()), row_stride);
    }
    
    file.close();
    std::cout << "Saved: " << filename << std::endl;
}

// ===== 協助函數：將數據保存為 CSV 文件 =====
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
    if (n == 0) {
        std::cerr << "Error: Modulo by zero (n=0)." << std::endl;
        return 0; 
    }
    if (n < 0) n = -n; // Ensure n is positive for modulo
    int m = x % n;
    return (m < 0) ? m + n : m;
}

// Box-Muller 變換生成高斯隨機數
float gaussianNoise() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1.0f); // Mean 0, stddev 1
    return dist(gen);
}

// ===== FFT 實現 (假設不變) =====
using Complex = std::complex<float>;
void fft1D(std::vector<Complex>& a, bool invert); // Declaration
void fft2D(const std::vector<std::vector<float>>& input, std::vector<std::vector<Complex>>& output); // Declaration
void saveSpectrum(const std::string& filename, const std::vector<std::vector<Complex>>& spectrum); // Declaration


// ===== 協助函數：計算統計數據 =====
struct DataStats {
    float avg = 0.0f;
    float var = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    long long count_nan_inf = 0;
};

DataStats calculateStats(const float* data, long long total_size, const std::string& name) {
    DataStats stats;
    if (total_size == 0) {
        std::cerr << "Warning: Calculating stats for zero-sized data: " << name << std::endl;
        return stats;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    long long valid_count = 0;

    for (long long i = 0; i < total_size; ++i) {
        float val = data[i];
        if (std::isnan(val) || std::isinf(val)) {
            stats.count_nan_inf++;
            continue;
        }
        valid_count++;
        sum += val;
        sum_sq += static_cast<double>(val) * val; // Use double for sum_sq to avoid precision loss
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
        }

    if (valid_count > 0) {
        stats.avg = static_cast<float>(sum / valid_count);
        // Variance = E[X^2] - (E[X])^2
        stats.var = static_cast<float>((sum_sq / valid_count) - static_cast<double>(stats.avg) * stats.avg);
    } else {
        std::cerr << "Warning: No valid data points to calculate stats for: " << name << std::endl;
        stats.min_val = 0; // Default if no valid points
        stats.max_val = 0;
    }


    std::cout << name << " stats: avg=" << stats.avg
              << " var=" << stats.var
              << " stddev=" << (stats.var > 0 ? sqrt(stats.var) : 0) // stddev is sqrt(variance)
              << " min=" << stats.min_val
              << " max=" << stats.max_val;
    if (stats.count_nan_inf > 0) {
        std::cout << " (NaN/Inf count: " << stats.count_nan_inf << ")";
        }
    std::cout << std::endl;
    return stats;
}


// ===== 協助函數：從 3D tile 中提取一個 2D XY 切片並保存為 BMP =====
void saveTileSlice(const std::string& filename_prefix, int slice_z, const float* tile_data, int n) {
    if (!tile_data) {
        std::cerr << "Error: tile_data is null for saveTileSlice: " << filename_prefix << std::endl;
        return;
    }
    if (n <= 0) {
        std::cerr << "Error: n=" << n << " is invalid for saveTileSlice: " << filename_prefix << std::endl;
        return;
    }
    if (slice_z < 0 || slice_z >= n) {
        std::cerr << "Error: Slice_z=" << slice_z <<" out of bounds [0," << n-1 << "] for saveTileSlice: " << filename_prefix << std::endl;
        return;
    }

    std::vector<std::vector<float>> slice_image(n, std::vector<float>(n));
    float current_min_val = std::numeric_limits<float>::max();
    float current_max_val = std::numeric_limits<float>::lowest();
    bool all_same = true;
    float first_val = 0.0f; // Placeholder if all_same is true

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            long long index = static_cast<long long>(x) + static_cast<long long>(y) * n + static_cast<long long>(slice_z) * n * n;
            // Bounds check for safety, though slice_z check should cover most
            if (index >= static_cast<long long>(n)*n*n) {
                 std::cerr << "Error: Index out of bounds in saveTileSlice inner loop." << std::endl;
                 slice_image[y][x] = 0.0f; // Default error value
                 continue;
            }
            float val = tile_data[index];
            slice_image[y][x] = val;

            if (std::isnan(val) || std::isinf(val)) continue; // Skip NaN/Inf for min/max calculation

            if (y == 0 && x == 0) first_val = val;
            else if (std::abs(val - first_val) > 1e-6f) all_same = false;

            if (val < current_min_val) current_min_val = val;
            if (val > current_max_val) current_max_val = val;
        }
    }
    
    // Normalize for visualization if there's a valid range
    if (!all_same && (current_max_val - current_min_val > 1e-6f) && std::isfinite(current_min_val) && std::isfinite(current_max_val) ) {
        for(int y=0; y<n; ++y) for(int x=0; x<n; ++x) {
            if (std::isfinite(slice_image[y][x])) { // Only normalize finite values
                 slice_image[y][x] = 2.0f * (slice_image[y][x] - current_min_val) / (current_max_val - current_min_val) - 1.0f;
            } else {
                 slice_image[y][x] = 0.0f; // Set NaN/Inf to neutral for image
            }
        }
    } else { // If all values are same or range is too small, or non-finite min/max
         for(int y=0; y<n; ++y) for(int x=0; x<n; ++x) {
            slice_image[y][x] = 0.0f; // Center value if all same or problematic range
         }
    }

    saveBMP(filename_prefix + "_slice_z" + std::to_string(slice_z) + ".bmp", slice_image);

    // Also save spectrum
    std::vector<std::vector<Complex>> spectrum_data; // Renamed to avoid conflict
    if (n > 0 && (n & (n - 1)) == 0) { 
        fft2D(slice_image, spectrum_data);
        saveSpectrum(filename_prefix + "_slice_z" + std::to_string(slice_z) + "_spectrum.bmp", spectrum_data);
    } else if (n > 0) { // Only print if n > 0
        std::cout << "Skipping spectrum for " << filename_prefix << " because size " << n << " is not a power of 2." << std::endl;
    }
}

// ===== Wavelet Noise 實現 =====
class WaveletNoise {
private:
    static constexpr int ARAD = 16;
    
public:
    static float* noiseTileData;
    static int noiseTileSize;

    static void GenerateNoiseTile(int n, int /*olap*/ = 0) {
        if (n <= 0) {
            std::cerr << "Error: GenerateNoiseTile called with n=" << n << ". Must be > 0." << std::endl;
            noiseTileData = nullptr;
            noiseTileSize = 0;
            return;
        }
        if (n % 2) n++; 
        noiseTileSize = n; 

        long long num_elements = static_cast<long long>(n) * n * n;
        size_t sz_bytes = num_elements * sizeof(float);

        float* temp1 = (float*)malloc(sz_bytes);
        float* temp2 = (float*)malloc(sz_bytes); 
        float* r_initial = (float*)malloc(sz_bytes); 
        float* noise_buffer = (float*)malloc(sz_bytes); // Renamed from 'noise' to avoid confusion with class member

        if (!temp1 || !temp2 || !r_initial || !noise_buffer) {
            std::cerr << "Error: Memory allocation failed in GenerateNoiseTile." << std::endl;
            free(temp1); free(temp2); free(r_initial); free(noise_buffer);
            noiseTileData = nullptr; noiseTileSize = 0;
            return;
        }
        
        // Step 1. Fill with random numbers (R)
        for (long long i = 0; i < num_elements; i++) {
            r_initial[i] = gaussianNoise();
        }
        std::cout << "--- Stats for R_initial ---" << std::endl;
        calculateStats(r_initial, num_elements, "R_initial");
        saveTileSlice("debug_R_initial", n / 2, r_initial, n);

        memcpy(noise_buffer, r_initial, sz_bytes); // Start with R for filtering
        
        // Steps 2 and 3. Downsample and upsample (R -> R_du)
        // Result of filtering will be in temp2
        // First pass (X): input=noise_buffer (R), intermediate=temp1, output=temp2
        for (int iy = 0; iy < n; iy++) {
            for (int iz = 0; iz < n; iz++) {
                long long base_idx = static_cast<long long>(iy) * n + static_cast<long long>(iz) * n * n;
                Downsample(&noise_buffer[base_idx], &temp1[base_idx], n, 1);
                Upsample(&temp1[base_idx], &temp2[base_idx], n, 1);
            }
        }
        // Second pass (Y): input=temp2, intermediate=temp1, output=temp2 (overwrite)
        for (int ix = 0; ix < n; ix++) {
            for (int iz = 0; iz < n; iz++) {
                long long base_idx = static_cast<long long>(ix) + static_cast<long long>(iz) * n * n;
                Downsample(&temp2[base_idx], &temp1[base_idx], n, n); 
                Upsample(&temp1[base_idx], &temp2[base_idx], n, n);
            }
        }
        // Third pass (Z): input=temp2, intermediate=temp1, output=temp2 (overwrite)
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                long long base_idx = static_cast<long long>(ix) + static_cast<long long>(iy) * n;
                Downsample(&temp2[base_idx], &temp1[base_idx], n, n * n);
                Upsample(&temp1[base_idx], &temp2[base_idx], n, n * n);
            }
        }
        // temp2 now holds R_down_up
        std::cout << "--- Stats for R_down_up (in temp2) ---" << std::endl;
        calculateStats(temp2, num_elements, "R_down_up");
        saveTileSlice("debug_R_down_up", n / 2, temp2, n);

        // Step 4. Subtract (N = R - R_down_up)
        // noise_buffer will hold N
        for (long long i = 0; i < num_elements; i++) {
            noise_buffer[i] = r_initial[i] - temp2[i]; 
        }
        std::cout << "--- Stats for N_after_subtract (in noise_buffer) ---" << std::endl;
        calculateStats(noise_buffer, num_elements, "N_after_subtract");
        saveTileSlice("debug_N_after_subtract", n / 2, noise_buffer, n);
        
        // Avoid even/odd variance difference
        int offset_val = n / 2; 
        if (offset_val % 2 == 0) offset_val++;
        
        // Use temp1 as temporary storage for N_offset
        for (int ix = 0; ix < n; ix++) {
            for (int iy = 0; iy < n; iy++) {
                for (int iz = 0; iz < n; iz++) {
                    long long target_idx = static_cast<long long>(ix) + static_cast<long long>(iy) * n + static_cast<long long>(iz) * n * n;
                    long long source_idx = static_cast<long long>(Mod(ix + offset_val, n)) + 
                                           static_cast<long long>(Mod(iy + offset_val, n)) * n + 
                                           static_cast<long long>(Mod(iz + offset_val, n)) * n * n;
                    if (source_idx < 0 || source_idx >= num_elements) { // Safety check
                        std::cerr << "Error: N_offset source_idx out of bounds." << std::endl;
                        temp1[target_idx] = 0.0f;
                    } else {
                         temp1[target_idx] = noise_buffer[source_idx];
                    }
                }
            }
        }
        
        for (long long i = 0; i < num_elements; i++) {
            noise_buffer[i] += temp1[i]; // N_final = N + N_offset
        }
        std::cout << "--- Stats for N_final (in noise_buffer) ---" << std::endl;
        calculateStats(noise_buffer, num_elements, "N_final");
        saveTileSlice("debug_N_final", n / 2, noise_buffer, n);

        free(r_initial);
        free(temp1);
        free(temp2);
        noiseTileData = noise_buffer; // noiseTileData now owns noise_buffer
    }
    
    static void Downsample(float* from, float* to, int n, int stride) {
        if (!from || !to || n <= 0 || stride <= 0) {
            std::cerr << "Error: Invalid args to Downsample. n=" << n << ", stride=" << stride << std::endl;
            // Consider filling 'to' with zeros here to prevent uninitialized data propagation
            if (to && n > 0 && (n/2 > 0)) { // n/2 output elements
                for(int i_out = 0; i_out < n/2; ++i_out) {
                    to[static_cast<long long>(i_out) * stride] = 0.0f;
                }
            }
            return;
        }

        // ARAD is 16. aCoeffs has 2*ARAD = 32 elements.
        // Coefficients should use 'f' suffix for float literals.
        float aCoeffs[2*ARAD] = {
            0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
            -0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
            0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
            0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
        };
        // a_centered points to aCoeffs[ARAD] (i.e., aCoeffs[16]).
        // Valid relative indices for a_centered: -ARAD to ARAD-1 (i.e., -16 to 15).
        // These access aCoeffs[0] to aCoeffs[2*ARAD-1] (i.e., aCoeffs[0] to aCoeffs[31]).
        float* a_centered = &aCoeffs[ARAD];

        for (int i_out = 0; i_out < n / 2; i_out++) { // For each output element 'to[i_out]'
            long long to_idx = static_cast<long long>(i_out) * stride;
            to[to_idx] = 0.0f; // Initialize output element

            // The filter tap index for 'a_centered' should range from -ARAD to ARAD-1.
            // This loop runs 2*ARAD times.
            for (int k_filter_tap_idx = -ARAD; k_filter_tap_idx < ARAD; ++k_filter_tap_idx) {
                // 'k_input_original_idx' is the corresponding index in the 'from' array space (like 'k' in paper's formula).
                // The formula is effectively: to[i_out] = sum_{k_filter_tap_idx} a_centered[k_filter_tap_idx] * from[2*i_out + k_filter_tap_idx]
                int k_input_original_idx = 2 * i_out + k_filter_tap_idx;

                long long from_idx_in_dimension = Mod(k_input_original_idx, n);
                long long from_element_absolute_idx = from_idx_in_dimension * stride;

                // Accessing a_centered[k_filter_tap_idx] is now safe.
                // Accessing from[from_element_absolute_idx] is safe due to Mod(..., n)
                // (assuming 'from' buffer is allocated correctly for 'n' elements along the dimension with 'stride').
                to[to_idx] += a_centered[k_filter_tap_idx] * from[from_element_absolute_idx];
            }
        }
    }
    
    static void Upsample(float* from, float* to, int n, int stride) {
        // ... (Upsample implementation as before, ensure 'from' and 'to' are valid)
        if (!from || !to || n <= 0 || stride <= 0) {
            std::cerr << "Error: Invalid args to Upsample." << std::endl;
            return;
        }
        float pCoeffs[4] = { 0.25, 0.75, 0.75, 0.25 };
        float* p = &pCoeffs[2]; // Center of pCoeffs for p[i-2k] indexing
        for (int i = 0; i < n; i++) {
            to[static_cast<long long>(i)*stride] = 0;
            // Loop k from i/2 to i/2 + 1 (inclusive)
            // This means k will take on two values for each i.
            // Example: i=0 -> k=0,1. i=1 -> k=0,1. i=2 -> k=1,2. i=3 -> k=1,2
            for (int k_offset = 0; k_offset <= 1; ++k_offset) {
                int k = i/2 + k_offset;
                int p_idx = i - 2*k; // This will be in range [-2, 1] typically, used to index pCoeffs around its center

                // Ensure p_idx is valid for p (which is &pCoeffs[2])
                // p[p_idx] means pCoeffs[2 + p_idx]
                // Valid indices for pCoeffs are 0,1,2,3
                // So 2+p_idx must be in [0,3] -> p_idx must be in [-2, 1]
                if (p_idx < -2 || p_idx > 1) { // Corresponds to pCoeffs indices outside 0..3
                     // This should not happen with correct k loop, but good for sanity
                    // std::cerr << "Upsample p_idx out of bounds: " << p_idx << std::endl;
                    continue;
                }
                long long from_idx = static_cast<long long>(Mod(k, n/2)) * stride;

                to[static_cast<long long>(i)*stride] += p[p_idx] * from[from_idx];
            }
        }
    }
    
    static float WNoise(float p[3]) {
        // ... (WNoise implementation as before, check noiseTileSize and noiseTileData)
        int n = noiseTileSize;
        if (n == 0 || noiseTileData == nullptr) {
            // std::cerr << "Error: WNoise called with uninitialized tile (size=" << n << ")." << std::endl;
            return 0.0f; 
        }
        int i, f[3], mid[3];
        float w[3][3], t, result = 0;
        for (i = 0; i < 3; i++) {
            mid[i] = ceil(p[i] - 0.5f); // ensure float literal
            t = mid[i] - (p[i] - 0.5f);
            w[i][0] = t * t / 2.0f;
            w[i][2] = (1.0f - t) * (1.0f - t) / 2.0f;
            w[i][1] = 1.0f - w[i][0] - w[i][2];
        }
        for (f[2] = -1; f[2] <= 1; f[2]++) {
            for (f[1] = -1; f[1] <= 1; f[1]++) {
                for (f[0] = -1; f[0] <= 1; f[0]++) {
                    float weight = 1.0f;
                    long long N_sq = static_cast<long long>(n)*n;
                    long long current_c[3]; // Use long long for intermediate index calculation
                    for (i = 0; i < 3; i++) {
                        current_c[i] = Mod(mid[i] + f[i], n);
                        weight *= w[i][f[i] + 1];
                    }
                    long long final_idx = current_c[0] + current_c[1] * n + current_c[2] * N_sq;
                    if (final_idx < 0 || final_idx >= N_sq * n) { // Safety check
                        // std::cerr << "Error: WNoise final_idx out of bounds." << std::endl;
                        continue;
                    }
                    result += weight * noiseTileData[final_idx];
                }
            }
        }
        
        return result;
    }
    
    // ... (WProjectedNoise, MultibandNoise, Noise2D, MultibandNoise2D as before) ...
    // 3D noise projected onto 2D (論文附錄2)
    static float WProjectedNoise(float p[3], float normal[3]) {
        int n = noiseTileSize;
         if (n == 0 || noiseTileData == nullptr) return 0.0f;
        int i, c[3], min_coord[3], max_coord[3]; // Renamed min/max to avoid conflict
        float support, result = 0;
        
        for (i = 0; i < 3; i++) {
            support = 3.0f * std::abs(normal[i]) + 3.0f * std::sqrt((1.0f - normal[i] * normal[i]) / 2.0f);
            min_coord[i] = static_cast<int>(std::ceil(p[i] - support));
            max_coord[i] = static_cast<int>(std::floor(p[i] + support));
        }
        
        for (c[2] = min_coord[2]; c[2] <= max_coord[2]; c[2]++) {
            for (c[1] = min_coord[1]; c[1] <= max_coord[1]; c[1]++) {
                for (c[0] = min_coord[0]; c[0] <= max_coord[0]; c[0]++) {
                    float t, t1, t2, t3, dot = 0, weight = 1;
                    for (i = 0; i < 3; i++) {
                        dot += normal[i] * (p[i] - c[i]);
                    }
                    for (i = 0; i < 3; i++) {
                        t = (c[i] + normal[i] * dot / 2.0f) - (p[i] - 1.5f);
                        t1 = t - 1.0f;
                        t2 = 2.0f - t;
                        t3 = 3.0f - t;
                        weight *= (t <= 0 || t >= 3.0f) ? 0.0f : 
                                 (t < 1.0f) ? t * t / 2.0f : 
                                 (t < 2.0f) ? 1.0f - (t1 * t1 + t2 * t2) / 2.0f : 
                                 t3 * t3 / 2.0f;
                    }
                    long long N_sq = static_cast<long long>(n)*n;
                    result += weight * noiseTileData[Mod(c[2], n) * N_sq + Mod(c[1], n) * n + Mod(c[0], n)];
                }
            }
        }
        return result;
    }
    
    // Multiband noise (論文附錄2)
    static float WMultibandNoise(float p_world[3], float s, float* normal, 
                                int firstBand, int nbands, float* w_weights) { // Renamed p to p_world, w to w_weights
        if (noiseTileSize == 0 || noiseTileData == nullptr) return 0.0f;
        float q_param[3], result = 0, variance_sum = 0; // Renamed q to q_param, variance to variance_sum
        int i, b;
        
        // Corrected loop condition and q calculation based on N(scale*p_world)
        for (b = 0; b < nbands; b++) {
            int current_band_j = firstBand + b;
            if (s + current_band_j >= 0 && nbands > 1) { // Only skip if s makes j too high, and more than one band requested
                 // For a single band request (nbands=1), always compute it regardless of j
                 // This aligns with how generateSingleBandDiagnostic calls WNoise directly
                 if (nbands == 1 && current_band_j >=0 ) {} // Allow if nbands=1
                 else continue; 
            }

            float scale = std::pow(2.0f, current_band_j);
            for (i = 0; i <= 2; i++) {
                q_param[i] = p_world[i] * scale; // N(scale * p_world) -> pass (scale * p_world) to WNoise
            }
            
            float band_noise = (normal) ? w_weights[b] * WProjectedNoise(q_param, normal) 
                                       : w_weights[b] * WNoise(q_param);
            if (std::isfinite(band_noise)) { // Add only finite contributions
                result += band_noise;
            }
        }
        
        for (b = 0; b < nbands; b++) {
            variance_sum += w_weights[b] * w_weights[b];
        }
        
        if (variance_sum > 1e-9f) { // Avoid division by zero or very small variance
            float norm_factor_sq = variance_sum * ((normal) ? 0.296f : 0.210f); //論文提的常數
            if (norm_factor_sq > 1e-9f) {
                 if (std::isfinite(result)) { // Only normalize if result is finite
                    result /= std::sqrt(norm_factor_sq);
                 } else {
                    // std::cerr << "Warning: Non-finite result before variance normalization." << std::endl;
                    result = 0.0f; // Reset if result became non-finite
                 }
            }
        } else if (nbands > 0 && variance_sum <= 1e-9f) {
            // If variance is zero (e.g. all weights are zero), result should be zero
            result = 0.0f;
        }
        
        return result;
    }
    
    // 2D convenience functions
    static float Noise2D(float x, float y) {
        float p[3] = {x, y, 0.5f}; // z can be any constant for 2D
        return WNoise(p);
    }
    
    static float MultibandNoise2D(float x, float y, int nbands, float* weights) {
        float p[3] = {x, y, 0.5f};
         // For typical fractal noise, firstBand is often 0 or negative.
         // If bands are 0, 1, 2, ..., then firstBand = 0.
         // If bands are ..., -2, -1, 0, then firstBand = some_negative_value.
         // Paper's M(x) = sum w_b N(2^b x), where b=b_min...b_max.
         // If we want b to go from 0 to nbands-1 for simplicity in weights array:
         // firstBand = 0, s = 0. Condition becomes b < 0 (incorrect for positive b)
         // Let's assume firstBand refers to the exponent of 2 for the coarsest band.
         // If nbands = 4, we want bands 2^0, 2^1, 2^2, 2^3 (or 2^-3, 2^-2, 2^-1, 2^0)
         // The condition s + firstBand + b < 0 is key.
         // If s=0, firstBand+b < 0.
         // For fractal noise where frequency doubles: bands j, j+1, j+2...
         // Let firstBand be the 'j' of the coarsest (lowest frequency) band we care about.
         // Example: if we want 4 octaves, starting from frequency 1 (j=0 for N(2^0 x))
         // then bands are j=0,1,2,3. firstBand=0. Loop b=0..3. (0+0+b < 0) fails.
         // The example in Appendix 2 WMultibandNoise has:
         // for (b=0; b < nbands && s+firstBand+b < 0; b++)
         // This implies that (firstBand + b) should generally be negative.
         // If we want bands N(x), N(2x), N(4x) ... these correspond to j=0, 1, 2...
         // If we want bands N(x), N(x/2), N(x/4) ... these correspond to j=0, -1, -2...
         // Let's use the common fractal noise setup:
         // Octave 0: freq 1 (scale_exp 0)
         // Octave 1: freq 2 (scale_exp 1)
         // Octave 2: freq 4 (scale_exp 2)
         // ...
         // The loop in WMultibandNoise implies j values should be < -s.
         // If s=0, j values must be negative.
         // To use positive exponents (higher frequencies), we need to adjust how `firstBand` is interpreted or `s` used.
         // Or, the paper's `N(2^j x)` means `j` is the resolution level.
         // `j=-1` is representable, `j>=0` is not.
         // If our "band 0" corresponds to `j=-nbands+1` (most detailed, highest freq if `j` increases for higher freq)
         // and "band nbands-1" corresponds to `j=0` (coarsest representable detail if `j` increases for lower freq).

        // Let's try to match typical fractal noise where frequency increases.
        // Band 0: scale 2^0. Band 1: scale 2^1, etc.
        // The condition `s + firstBand + b < 0` means that `firstBand + b` should be negative.
        // To generate `N(p), N(2p), N(4p)` etc. `firstBand` should be negative such that `firstBand+b` covers these.
        // This is confusing. Let's assume the WMultibandNoise is meant for negative j exponents primarily for now.
        // For a simple fractal sum (fBm like):
        // result = 0; float freq = 1.0; float amp = 1.0;
        // for (int i=0; i<nbands; ++i) { result += amp * Noise2D(x*freq, y*freq); freq *= 2.0; amp *= persistence; }
        // This does not use the WMultibandNoise structure directly.

        // If we follow the paper's Figure 10(a) "12 bands with Gaussian distribution"
        // and 10(b) "8 bands with white distribution". These are sums of N(2^j x).
        // The condition `s+firstBand+b < 0` implies that the `j` values (which is `firstBand+b`)
        // should be negative. This corresponds to bands N(x/2), N(x/4), etc.
        // So, `firstBand` should be something like `-nbands` or `-nbands+1` to make `firstBand+b` negative.

        return WMultibandNoise(p, 0.0f, nullptr, -nbands, nbands, weights); // Try firstBand = -nbands

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

// ===== 測試函數 (generateSingleBandDiagnostic 假設不變) =====
void generateSingleBandDiagnostic(); // Declaration


// --- Implementations for FFT and Test Functions (if not in separate .cpp) ---
// (Copied from your provided code for completeness in a single block)

// 1D Cooley-Tukey FFT (Radix-2)
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

// 快速 2D FFT 實現
void fft2D(const std::vector<std::vector<float>>& input,
                std::vector<std::vector<Complex>>& output) {
    if (input.empty() || input[0].empty()) {
        std::cerr << "FFT Error: Input data is empty." << std::endl;
        output.clear();
        return;
    }
    int height = input.size();
    int width = input[0].size();
    
    if ((height & (height - 1)) != 0 || (width & (width - 1)) != 0 || height == 0 || width == 0) {
        std::cerr << "FFT Error: Image dimensions (" << width << "x" << height << ") must be non-zero and powers of 2." << std::endl;
        output.assign(height, std::vector<Complex>(width, 0)); // Assign to keep size consistent for caller
        return;
    }

    output.assign(height, std::vector<Complex>(width)); // Use assign for clarity
    for(int i=0; i<height; ++i) {
        for(int j=0; j<width; ++j) {
            if (std::isfinite(input[i][j])) {
                output[i][j] = input[i][j];
            } else {
                // std::cerr << "Warning: Non-finite input to FFT at (" << j << "," << i << "). Using 0." << std::endl;
                output[i][j] = 0.0f; // Replace NaN/Inf with 0 for FFT
            }
        }
    }

    for (int i = 0; i < height; i++) {
        fft1D(output[i], false);
    }

    std::vector<Complex> col(height);
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            col[i] = output[i][j];
        }
        fft1D(col, false);
        for (int i = 0; i < height; i++) {
            output[i][j] = col[i];
        }
    }
}

void saveSpectrum(const std::string& filename, const std::vector<std::vector<Complex>>& spectrum) {
    if (spectrum.empty() || spectrum[0].empty()) {
        std::cerr << "Error: Empty spectrum data for saveSpectrum: " << filename << std::endl;
        return;
    }
    int height = spectrum.size();
    int width = spectrum[0].size();

    // Step 1: Calculate magnitudes and find maxMag
    std::vector<std::vector<float>> magnitudes(height, std::vector<float>(width));
    float maxMag = 0.0f;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = std::abs(spectrum[i][j]);
            if (std::isfinite(mag)) {
                magnitudes[i][j] = mag;
                if (mag > maxMag) maxMag = mag;
            } else {
                magnitudes[i][j] = 0.0f;
            }
        }
    }

    // Step 2: Create fft-shifted magnitudes and save to CSV
    std::vector<std::vector<float>> shifted_magnitudes(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int shiftedI = (i + height / 2) % height;
            int shiftedJ = (j + width / 2) % width;
            shifted_magnitudes[i][j] = magnitudes[shiftedI][shiftedJ];
        }
    }

    // Save shifted magnitudes to CSV
    std::string csv_filename = filename;
    size_t dot_pos = csv_filename.rfind(".bmp");
    if (dot_pos != std::string::npos) {
        csv_filename.replace(dot_pos, 4, "_magnitude.csv");
    } else {
        csv_filename += "_magnitude.csv";
    }
    saveDataAsCSV(csv_filename, shifted_magnitudes);


    // Step 3: Calculate log-magnitudes from shifted data for BMP visualization
    std::vector<std::vector<float>> log_magnitudes(height, std::vector<float>(width));
    float min_log_m = std::numeric_limits<float>::max();
    float max_log_m = std::numeric_limits<float>::lowest();
    bool first_finite_log_m = true;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag_val = shifted_magnitudes[i][j];
            float current_log_mag;
            if (maxMag > 1e-9f) {
                 // 增大 FACTOR 可以讓低幅值部分在 log 後有更大的相對值
                 current_log_mag = std::log(1.0f + mag_val / maxMag * 1000.0f); // 嘗試更大的 FACTOR, e.g., 1000, 10000
            } else {
                current_log_mag = 0.0f; // log(1) is 0
            }
            log_magnitudes[i][j] = current_log_mag;
            if (std::isfinite(current_log_mag)) {
                if (first_finite_log_m || current_log_mag < min_log_m) min_log_m = current_log_mag;
                if (first_finite_log_m || current_log_mag > max_log_m) max_log_m = current_log_mag;
                first_finite_log_m = false;
            }
        }
    }

    if (first_finite_log_m) { // All log_magnitudes were NaN/Inf or empty
        min_log_m = 0.0f;
        max_log_m = 1.0f; // Arbitrary small positive range if all else fails
    }
    if (std::abs(max_log_m - min_log_m) < 1e-6f) { // If range is still too small, expand it slightly
        max_log_m = min_log_m + 1.0f; // Ensure a non-zero range for division
    }


    // Step 4: Re-normalize log_magnitudes to [-1, 1] for BMP output
    std::vector<std::vector<float>> output_img(height, std::vector<float>(width));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (std::isfinite(log_magnitudes[i][j])) {
                if (max_log_m - min_log_m > 1e-6f) { // Check for valid range before division
                    output_img[i][j] = 2.0f * (log_magnitudes[i][j] - min_log_m) / (max_log_m - min_log_m) - 1.0f;
                } else { // Range is too small, map to a mid-gray or black
                    output_img[i][j] = -1.0f; // Or 0.0f for mid-gray
                }
            } else {
                output_img[i][j] = -1.0f; // Black for NaN/Inf in log_magnitudes
            }
        }
    }
    saveBMP(filename, output_img);
}

void generateSingleBandDiagnostic() {
    std::cout << "Generating single band diagnostic images..." << std::endl;
    if (WaveletNoise::noiseTileSize == 0 || WaveletNoise::noiseTileData == nullptr) {
        std::cerr << "Error in generateSingleBandDiagnostic: Noise tile not initialized." << std::endl;
        return;
    }
    
    int imageSize = 128; 
    if ((imageSize & (imageSize-1)) != 0) { // Ensure imageSize is power of 2 for FFT
        std::cerr << "Warning: diagnostic imageSize " << imageSize << " is not a power of 2. FFT might fail." << std::endl;
        // Optionally, find next power of 2 or adjust
    }
    
    std::vector<std::vector<float>> baseBand(imageSize, std::vector<float>(imageSize));
    int tileSize = WaveletNoise::noiseTileSize;
    
    // Combo 1: imageSize = tileSize, multiplier = 1
    imageSize = tileSize; // Override for Combo 1
    std::cout << "Using Combo 1: imageSize=" << imageSize << ", tileSize=" << tileSize << ", multiplier=1" << std::endl;
    baseBand.assign(imageSize, std::vector<float>(imageSize)); // Resize

    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            float fx = (float)x / imageSize * tileSize * 1.0f; 
            float fy = (float)y / imageSize * tileSize * 1.0f;
            float p[3] = {fx, fy, 0.5f};
            baseBand[y][x] = WaveletNoise::WNoise(p);
        }
    }
    saveBMP("diagnostic_base_band.bmp", baseBand);
    std::vector<std::vector<Complex>> baseSpectrum;
    fft2D(baseBand, baseSpectrum);
    saveSpectrum("diagnostic_base_band_spectrum.bmp", baseSpectrum);
    
    // Scaled bands (optional, can be run after base band is confirmed good)
    // Ensure imageSize for scaled bands is also power of 2
    imageSize = 128; // Reset for scaled bands or use a consistent power of 2
    std::cout << "Generating scaled bands with imageSize=" << imageSize << std::endl;

    for (int scale_exp = -1; scale_exp <= 1; scale_exp++) { // Reduced range for quicker test
        std::vector<std::vector<float>> scaledBand(imageSize, std::vector<float>(imageSize));
        float scale = std::pow(2.0f, scale_exp);
        
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                // For scaled bands, the 'tileSize' in fx,fy should probably be fixed 
                // to a reference scale, or fx,fy should be directly scaled world coords.
                // Let's use world coords directly: map image to [0, world_extent * scale]
                float world_extent = static_cast<float>(WaveletNoise::noiseTileSize); // Or some other reference like 16.0f
                float fx = (float)x / imageSize * world_extent * scale; 
                float fy = (float)y / imageSize * world_extent * scale;
                float p[3] = {fx, fy, 0.5f};
                scaledBand[y][x] = WaveletNoise::WNoise(p);
            }
        }
        saveBMP("diagnostic_band_scale_exp" + std::to_string(scale_exp) + ".bmp", scaledBand);
        std::vector<std::vector<Complex>> scaledSpectrum;
        fft2D(scaledBand, scaledSpectrum);
        saveSpectrum("diagnostic_spectrum_scale_exp" + std::to_string(scale_exp) + ".bmp", scaledSpectrum);
    }
}

void generateSpectralAnalysis(int imageSize = 256) {
    std::cout << "Performing spectral analysis..." << std::endl;
    
    std::vector<std::vector<float>> wavelet(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> perlin(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> waveletSlice(imageSize, std::vector<float>(imageSize));
    std::vector<std::vector<float>> waveletProjected(imageSize, std::vector<float>(imageSize));
    
    PerlinNoise perlinGen;
    float normal[3] = {0, 0, 1};
    
    // 修正：使用正確的座標範圍來展示帶限特性
    int tileSize = WaveletNoise::noiseTileSize;
    
    for (int y = 0; y < imageSize; y++) {
        for (int x = 0; x < imageSize; x++) {
            // 修正：座標映射到完整的 tile 範圍
            float fx = (float)x / imageSize * tileSize;
            float fy = (float)y / imageSize * tileSize;
            
            // 2D noise
            float p2d[3] = {fx, fy, 0.5f};
            wavelet[y][x] = WaveletNoise::WNoise(p2d);
            
            // Perlin noise (保持原始縮放用於比較)
            perlin[y][x] = perlinGen.noise2D(x / 32.0f, y / 32.0f);
            
            // 3D slice
            float p3d[3] = {fx, fy, 0.5f};
            waveletSlice[y][x] = WaveletNoise::WNoise(p3d);
            
            // 3D projected
            p3d[2] = 0.0f;
            waveletProjected[y][x] = WaveletNoise::WProjectedNoise(p3d, normal);
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
    
    // 生成 3 個相鄰頻帶，展示頻帶分離
    for (int band = 0; band < 3; band++) {
        std::vector<std::vector<float>> waveletBand(imageSize, std::vector<float>(imageSize));
        std::vector<std::vector<float>> perlinBand(imageSize, std::vector<float>(imageSize));
        
        PerlinNoise perlin;
        float scale = pow(2.0f, band);  // 2^0=1, 2^1=2, 2^2=4
        
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                // 修正：直接計算 N(2^band * x) 
                float evalX = (float)x / imageSize * scale;
                float evalY = (float)y / imageSize * scale;
                float p[3] = {evalX, evalY, 0.5f};
                
                // 直接使用 WNoise 評估單一頻帶
                waveletBand[y][x] = WaveletNoise::WNoise(p);
                
                // Perlin 對應頻帶
                perlinBand[y][x] = perlin.noise(evalX, evalY, 0.5f);
            }
        }
        
        saveBMP("wavelet_band_" + std::to_string(band) + ".bmp", waveletBand);
        saveBMP("perlin_band_" + std::to_string(band) + ".bmp", perlinBand);
        
        // Compute spectrum for each band
        std::vector<std::vector<Complex>> waveletSpec, perlinSpec;
        fft2D(waveletBand, waveletSpec);
        fft2D(perlinBand, perlinSpec);
        
        saveSpectrum("spectrum_wavelet_band_" + std::to_string(band) + ".bmp", waveletSpec);
        saveSpectrum("spectrum_perlin_band_" + std::to_string(band) + ".bmp", perlinSpec);
    }
}

// ===== 主程式 =====
int main() {
    // std::cout << "=== Wavelet Noise Complete Implementation ===" << std::endl;
    // std::cout << "Based on 'Wavelet Noise' by Cook & DeRose (2005)" << std::endl << std::endl;
    
    // // Initialize noise tile
    // std::cout << "Generating noise tile..." << std::endl;
    std::cout << "=== Wavelet Noise Diagnostic Mode ===" << std::endl;
    
    std::cout << "Generating noise tile (128x128x128)..." << std::endl;
    WaveletNoise::GenerateNoiseTile(128); 
    
    if (WaveletNoise::noiseTileData == nullptr || WaveletNoise::noiseTileSize == 0) {
        std::cerr << "Critical Error: Noise tile not generated or size is zero. Aborting." << std::endl;
        return 1;
    }
    
    // // Generate all comparisons with fixes
    // generateNoiseComparison();
    // generateSpectralAnalysis();  // 使用修正版本
    // generateBandComparison();     // 使用修正版本
    
    // 新增：診斷圖像
    std::cout << "\nStarting generateSingleBandDiagnostic()..." << std::endl;
    generateSingleBandDiagnostic();
    
    std::cout << "\n--- Finished ---" << std::endl;
    
    // std::cout << "\nGenerated files:" << std::endl;

    // std::cout << "\nVisual Comparison:" << std::endl;
    // std::cout << "- wavelet_2d.bmp / perlin_2d.bmp: 2D noise comparison" << std::endl;
    // std::cout << "- wavelet_3d_slice.bmp / perlin_3d_slice.bmp: 3D noise slice" << std::endl;
    // std::cout << "- wavelet_3d_projected.bmp: 3D noise projected to 2D" << std::endl;
    // std::cout << "- wavelet_multiband.bmp / perlin_multiband.bmp: Multiband noise" << std::endl;
    
    // std::cout << "\nSpectral Analysis:" << std::endl;
    // std::cout << "- spectrum_wavelet_2d.bmp / spectrum_perlin_2d.bmp: 2D noise spectra" << std::endl;
    // std::cout << "- spectrum_wavelet_3d_slice.bmp: 3D slice spectrum" << std::endl;
    // std::cout << "- spectrum_wavelet_3d_projected.bmp: 3D projected spectrum" << std::endl;
    
    // std::cout << "\nFrequency Band Analysis:" << std::endl;
    // std::cout << "- wavelet_band_0,1,2.bmp / perlin_band_0,1,2.bmp: Individual bands" << std::endl;
    // std::cout << "- spectrum_wavelet_band_0,1,2.bmp / spectrum_perlin_band_0,1,2.bmp: Band spectra" << std::endl;
    
    // std::cout << "\nDiagnostic files:" << std::endl;
    // std::cout << "- diagnostic_base_band.bmp: Base band N(x) over multiple tiles" << std::endl;
    // std::cout << "- diagnostic_base_band_spectrum.bmp: Spectrum showing band-limited nature" << std::endl;
    // std::cout << "- diagnostic_band_scale_*.bmp: Different frequency bands" << std::endl;
    // std::cout << "- diagnostic_spectrum_scale_*.bmp: Spectra of different bands" << std::endl;

    return 0;
}