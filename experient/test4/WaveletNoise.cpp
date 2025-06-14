#include "WaveletNoise.hpp"
#include "CommonUtils.hpp" // For random_float, b_spline_quadratic_eval
#include <cmath> // For std::floor
#include <iostream>
#include <numeric> // For std::accumulate

// CD05 Quadratic B-Spline Filter Coefficients (Appendix 1)
// Analysis filter a_k (for downsampling) - symmetric, ARAD = 16 (length 31)
const std::vector<float> WaveletNoise::aCoeffs = {
    0.000334f, -0.001528f,  0.000410f,  0.003545f, -0.000938f, -0.008233f,  0.002172f,  0.019120f,
   -0.005040f, -0.044412f,  0.011655f,  0.103311f, -0.025936f, -0.243780f,  0.033979f,  0.655340f,
    0.655340f,  0.033979f, -0.243780f, -0.025936f,  0.103311f,  0.011655f, -0.044412f, -0.005040f,
    0.019120f,  0.002172f, -0.008233f, -0.000938f,  0.003545f,  0.000410f, -0.001528f,  0.000334f
};
// Centered at index 15 (0.655340f)

// Synthesis filter p_k (for upsampling) - symmetric
const std::vector<float> WaveletNoise::pCoeffs = {0.25f, 0.75f, 0.75f, 0.25f}; 
// Centered at index 1 (0.75f) - p_k for k=-1,0,1,2 (relative to center)

WaveletNoise::WaveletNoise(int tileSize, unsigned int seed) : N_tile(tileSize) {
    if ((N_tile & (N_tile - 1)) != 0 || N_tile == 0) {
        throw std::invalid_argument("WaveletNoise tileSize must be a power of 2 and > 0.");
    }
    noise_coefficients.resize(N_tile, N_tile);
    generateTile(seed);
}


// 1D Convolution (can be row or column)
// data: input/output vector
// filter_coeffs: filter kernel
// downsample: if true, output is half size. if false (upsample), output is double size.
// circular: true for circular convolution
void WaveletNoise::convolve1D(std::vector<float>& data_in_out, const std::vector<float>& filter_coeffs, bool downsample_mode, bool circular) {
    int n_in = data_in_out.size();
    int n_out = downsample_mode ? (n_in / 2) : (n_in * 2); // For upsampling, input is already half size
    if (!downsample_mode) n_out = n_in; // Upsampling uses original size for output (input is half)


    std::vector<float> temp_out(n_out);
    
    int filter_center = (filter_coeffs.size() -1) / 2;

    for (int i_out = 0; i_out < n_out; ++i_out) {
        float sum = 0.0f;
        
        int i_in_center = downsample_mode ? (i_out * 2) : (i_out / 2); // Map output index to input center

        for (int k_filter = 0; k_filter < filter_coeffs.size(); ++k_filter) {
            int filter_offset = k_filter - filter_center;
            int i_in;

            if (downsample_mode) { // Downsampling: Sum over input samples
                i_in = i_in_center + filter_offset;
            } else { // Upsampling: Sum over "half-size" input, filter is applied to double-rate grid
                 // p_k where k = i_out - 2*j_in
                 // j_in = (i_out - k_filter_relative) / 2
                 if ((i_out - filter_offset) % 2 != 0 && n_in != n_out/2) { // n_in check for upsampling
                     // This filter tap doesn't align with an input sample for upsampling
                     // This logic is tricky, the paper's formula is better:
                     // f'_i = Sum_k p_{i-2k} f_k (for upsampling from f to f')
                     // Where f is half-size. Let's re-do upsampling logic.
                 }
                 i_in = (i_out + filter_offset); // THIS IS THE BUGGY PART FOR UPSAMPLING
                 // Correct for upsampling: Input `data_in_out` is `f_k`. We want `f'_i`.
                 // Filter coeff p_j is for tap j. We need p_{i_out - 2*k_in}.
                 // Iterate over k_in (input indices).
                 // For now, let's simplify and directly implement paper's summation.
            }

            // Simplified logic for convolution (assuming filter_coeffs is correctly indexed from 0)
            // This needs careful re-evaluation based on CD05's Appendix 1 formulation
            // For now, this is a generic convolution.
            // Upsampling: f'_i = Sum_k P_k f_{i/2 - k} (if i is even, 0 if odd, effectively)
            // Downsampling: f'_i = Sum_k A_k f_{2i - k} 

            // Simpler: CD05 Upsample/Downsample calls (from Appendix code)
            // to[i*stride] = 0; for (int k=i/2; k<=i/2+1; k++) to[i*stride] += p[i-2*k] * from[Mod(k,n/2)*stride]; // Upsample
            // to[i*stride] = 0; for (int k=2*i-ARAD; k<=2*i+ARAD; k++) to[i*stride] += a[k-2*i] * from[Mod(k,n)*stride]; // Downsample

            int current_data_idx = i_in_center + (k_filter - filter_center);
            if(downsample_mode) current_data_idx = i_out * 2 + (k_filter - filter_center); // Centering filter around 2*i_out
            else { // Upsampling
                // This means data_in_out is the coarse signal
                // We are calculating fine signal sample temp_out[i_out]
                // temp_out[i_out] = sum_j filter[i_out - 2*j] * data_in_out[j]
                // Let filter_idx = i_out - 2*j. filter_coeffs[filter_idx]
                // Here, k_filter is essentially j.
                // filter_relative_idx = i_out - 2 * (current_data_idx) <- wrong
            }


            if (circular) {
                current_data_idx = (current_data_idx % n_in + n_in) % n_in; // Ensure positive modulo
            }

            if (current_data_idx >= 0 && current_data_idx < n_in) {
                 sum += filter_coeffs[k_filter] * data_in_out[current_data_idx];
            }
        }
        temp_out[i_out] = sum;
    }
    // This convolve1D is too generic and likely incorrect for the specific CD05 filters.
    // Let's use the direct loop structure from CD05 Appendix 1 (adapted).

    data_in_out.assign(n_out, 0.0f); // Resize and zero out
    if (downsample_mode) { // Downsampling logic from CD05
        int ARAD_equiv = filter_center; // aCoeffs has center at index 15 for 31 elements
        for (int i = 0; i < n_out; ++i) { // n_out = n_in / 2
            float val = 0.0f;
            for (int k_offset = -ARAD_equiv; k_offset <= ARAD_equiv; ++k_offset) {
                // filter_coeffs index is k_offset + ARAD_equiv
                // data index is 2*i + k_offset
                int data_idx = (2 * i + k_offset);
                if (circular) data_idx = (data_idx % n_in + n_in) % n_in;
                
                if (data_idx >= 0 && data_idx < n_in) { // Check bounds if not circular
                    val += filter_coeffs[k_offset + ARAD_equiv] * temp_out[data_idx]; // temp_out holds original data here
                }
            }
            data_in_out[i] = val;
        }
    } else { // Upsampling logic from CD05
        // input data (temp_out) is n_in (coarse), output (data_in_out) is n_out = 2 * n_in (fine)
        // pCoeffs: p[-1]=0.25, p[0]=0.75, p[1]=0.75, p[2]=0.25
        // filter_center for pCoeffs (4 elements) is tricky. Let's say index 0 is p[-1].
        // filter_coeffs[0] = p[-1], filter_coeffs[1]=p[0], etc.
        int p_filter_len = filter_coeffs.size(); // Should be 4
        for (int i = 0; i < n_out; ++i) { // n_out = n_in * 2
            float val = 0.0f;
            // Iterate k_coarse from i/2 - 1 to i/2 (roughly) for pCoeffs
            for (int k_coarse_offset = 0; k_coarse_offset < n_in; ++k_coarse_offset) {
                // We need filter_coeffs[ i - 2*k_coarse_offset ]
                int p_idx = i - 2 * k_coarse_offset; // This is the p_k index like p_{-1}, p_0 etc.
                                                 // Map this to filter_coeffs index.
                                                 // If pCoeffs stores p[-1],p[0],p[1],p[2]
                                                 // then index is p_idx + 1.
                int p_coeffs_actual_idx = p_idx + 1; // Assuming pCoeffs[0] = p_{-1}

                if (p_coeffs_actual_idx >= 0 && p_coeffs_actual_idx < p_filter_len) {
                    int data_idx = k_coarse_offset;
                    if (circular) data_idx = (data_idx % n_in + n_in) % n_in;

                    if (data_idx >= 0 && data_idx < n_in) { // Check bounds if not circular
                         val += filter_coeffs[p_coeffs_actual_idx] * temp_out[data_idx]; // temp_out is original data
                    }
                }
            }
            data_in_out[i] = val;
        }
    }
}


void WaveletNoise::generateTile(unsigned int seed) {
    std::mt19937 tile_rng(seed);
    std::normal_distribution<float> gaussian_dist(0.0f, 1.0f); // For initial R

    Image R(N_tile, N_tile);
    for (int y = 0; y < N_tile; ++y) {
        for (int x = 0; x < N_tile; ++x) {
            R.at(x, y) = gaussian_dist(tile_rng);
        }
    }
    // std::cout << "Wavelet: Initial R generated." << std::endl;
    // R.savePPM("debug_R_initial.ppm");


    Image R_down_up = R; // Start with a copy of R

    // Temp storage for 1D operations
    std::vector<float> line_data(N_tile);
    std::vector<float> half_line_data(N_tile / 2);


    // --- Process Rows (Downsample then Upsample) ---
    // std::cout << "Wavelet: Processing rows..." << std::endl;
    for (int y = 0; y < N_tile; ++y) {
        // Copy row to line_data
        for (int x = 0; x < N_tile; ++x) line_data[x] = R_down_up.at(x, y);
        
        // Downsample (aCoeffs)
        std::vector<float> current_input_for_down = line_data;
        half_line_data.assign(N_tile/2, 0.0f); // Correct size for output of downsample
        int ARAD_a = (aCoeffs.size()-1)/2;
        for(int i_out = 0; i_out < N_tile/2; ++i_out) {
            float sum = 0.0f;
            for(int k_filt = 0; k_filt < aCoeffs.size(); ++k_filt) {
                int data_idx = (2*i_out + (k_filt - ARAD_a) + N_tile) % N_tile;
                sum += aCoeffs[k_filt] * current_input_for_down[data_idx];
            }
            half_line_data[i_out] = sum;
        }

        // Upsample (pCoeffs)
        line_data.assign(N_tile, 0.0f); // Correct size for output of upsample
        // pCoeffs effectively for indices -1, 0, 1, 2. Stored at 0, 1, 2, 3
        for(int i_out = 0; i_out < N_tile; ++i_out) {
            float sum = 0.0f;
            for(int k_coarse = 0; k_coarse < N_tile/2; ++k_coarse) {
                int p_idx_relative = i_out - 2*k_coarse; // e.g., -1, 0, 1, 2
                int p_coeff_array_idx = p_idx_relative + 1; // Map to 0,1,2,3 for pCoeffs
                if (p_coeff_array_idx >=0 && p_coeff_array_idx < pCoeffs.size()) {
                     sum += pCoeffs[p_coeff_array_idx] * half_line_data[k_coarse];
                }
            }
            line_data[i_out] = sum;
        }

        // Copy back to R_down_up
        for (int x = 0; x < N_tile; ++x) R_down_up.at(x, y) = line_data[x];
    }
    // std::cout << "Wavelet: Rows processed." << std::endl;
    // R_down_up.savePPM("debug_R_rows_processed.ppm");

    // --- Process Columns (Downsample then Upsample) ---
    // std::cout << "Wavelet: Processing columns..." << std::endl;
    for (int x = 0; x < N_tile; ++x) {
        // Copy column to line_data
        for (int y = 0; y < N_tile; ++y) line_data[y] = R_down_up.at(x, y);

        // Downsample
        std::vector<float> current_input_for_down = line_data;
        half_line_data.assign(N_tile/2, 0.0f);
        int ARAD_a = (aCoeffs.size()-1)/2;
        for(int i_out = 0; i_out < N_tile/2; ++i_out) {
            float sum = 0.0f;
            for(int k_filt = 0; k_filt < aCoeffs.size(); ++k_filt) {
                int data_idx = (2*i_out + (k_filt - ARAD_a) + N_tile) % N_tile;
                sum += aCoeffs[k_filt] * current_input_for_down[data_idx];
            }
            half_line_data[i_out] = sum;
        }
        
        // Upsample
        line_data.assign(N_tile, 0.0f);
        for(int i_out = 0; i_out < N_tile; ++i_out) {
            float sum = 0.0f;
            for(int k_coarse = 0; k_coarse < N_tile/2; ++k_coarse) {
                int p_idx_relative = i_out - 2*k_coarse;
                int p_coeff_array_idx = p_idx_relative + 1;
                if (p_coeff_array_idx >=0 && p_coeff_array_idx < pCoeffs.size()) {
                     sum += pCoeffs[p_coeff_array_idx] * half_line_data[k_coarse];
                }
            }
            line_data[i_out] = sum;
        }

        // Copy back to R_down_up
        for (int y = 0; y < N_tile; ++y) R_down_up.at(x, y) = line_data[y];
    }
    // std::cout << "Wavelet: Columns processed." << std::endl;
    // R_down_up.savePPM("debug_R_cols_processed_R_down_up.ppm");


    // --- Subtract to get noise coefficients ---
    // noise_coefficients_ij = R_ij - R_down_up_ij
    for (int y = 0; y < N_tile; ++y) {
        for (int x = 0; x < N_tile; ++x) {
            noise_coefficients.at(x, y) = R.at(x, y) - R_down_up.at(x, y);
        }
    }
    // Optional: Correct for even/odd variance (CD05 Section 4.3)
    // For simplicity, this is omitted here but involves adding an offset version of the tile to itself.

    // Verification: min/max of noise coefficients
    // float min_coeff = noise_coefficients.pixels[0], max_coeff = noise_coefficients.pixels[0];
    // for(float val : noise_coefficients.pixels) {
    //     if(val < min_coeff) min_coeff = val;
    //     if(val > max_coeff) max_coeff = val;
    // }
    // std::cout << "Wavelet Noise Tile Coefficients: min=" << min_coeff << ", max=" << max_coeff << std::endl;
    // noise_coefficients.savePPM("debug_N_coeffs.ppm", false, true);
}


float WaveletNoise::noise(float x_tile_units, float y_tile_units) const {
    // Evaluate N(x,y) = sum_ix,iy n_ix,iy B(2x-ix)B(2y-iy)
    // where x, y are evaluation points, ix, iy are coefficient indices
    // Let u = 2*x_tile_units, v = 2*y_tile_units
    
    float u = 2.0f * x_tile_units;
    float v = 2.0f * y_tile_units;

    int iu_base = static_cast<int>(std::floor(u));
    int iv_base = static_cast<int>(std::floor(v));

    float fu = u - static_cast<float>(iu_base); // fractional part for u, in [0,1)
    float fv = v - static_cast<float>(iv_base); // fractional part for v, in [0,1)

    float total_value = 0.0f;

    // The B-spline B(t) has support on [0,3).
    // So B(u - ix) is non-zero when 0 <= u - ix < 3  =>  u-3 < ix <= u
    // This means ix can be iu, iu-1, iu-2
    // Similar for iy: iv, iv-1, iv-2

    for (int dy = 0; dy < 3; ++dy) { // Corresponds to B(fv+2-dy), i.e. iy = iv-(2-dy)
        int iy_coeff = (iv_base - (2 - dy) + N_tile) % N_tile; // Coefficient index (circular)
        float b_val_y = b_spline_quadratic_eval(fv + static_cast<float>(2 - dy));

        if (b_val_y == 0.0f) continue;

        for (int dx = 0; dx < 3; ++dx) { // Corresponds to B(fu+2-dx), i.e. ix = iu-(2-dx)
            int ix_coeff = (iu_base - (2 - dx) + N_tile) % N_tile; // Coefficient index (circular)
            float b_val_x = b_spline_quadratic_eval(fu + static_cast<float>(2 - dx));
            
            if (b_val_x == 0.0f) continue;

            total_value += noise_coefficients.at(ix_coeff, iy_coeff) * b_val_x * b_val_y;
        }
    }
    return total_value;
}