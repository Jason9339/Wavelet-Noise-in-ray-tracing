#include "wavelet_noise_2d.h"
#include "rtweekend.h"
#include <iostream> // For debugging

// Initialize static const members (same as 3D version)
const std::vector<double> wavelet_noise_2d::aCoeffs = {
    0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002172, 0.019120,
    -0.005040, -0.044412, 0.011655, 0.103311, -0.025936, -0.243780, 0.033979, 0.655340,
    0.655340, 0.033979, -0.243780, -0.025936, 0.103311, 0.011655, -0.044412, -0.005040,
    0.019120, 0.002172, -0.008233, -0.000938, 0.003546, 0.000410, -0.001528, 0.000334
};
const std::vector<double> wavelet_noise_2d::pCoeffs = {0.25, 0.75, 0.75, 0.25};

int wavelet_noise_2d::Mod(int x, int n) {
    int m = x % n;
    return (m < 0) ? m + n : m;
}

double wavelet_noise_2d::gaussian_noise(std::mt19937& gen) {
    static std::normal_distribution<> d(0,1);
    return d(gen);
}

wavelet_noise_2d::wavelet_noise_2d(int tile_size_) {
    if (tile_size_ <= 0) tile_size_ = 32;
    if (tile_size_ % 2 != 0) tile_size_++;
    tile_size = tile_size_;
    GenerateNoiseTile2D(tile_size, 0);
}

// --- 1D Downsample/Upsample (simplified from 3D versions' structure) ---
void wavelet_noise_2d::Downsample1D(const std::vector<double>& from_data, std::vector<double>& to_data_half, int n_elements) const {
    to_data_half.assign(n_elements / 2, 0.0);
    for (int i = 0; i < n_elements / 2; ++i) {
        double sum = 0.0;
        for (int k_filter = 0; k_filter < 2 * ARAD; ++k_filter) {
            int from_relative_idx = 2 * i + (k_filter - ARAD);
            int from_actual_idx_in_slice = Mod(from_relative_idx, n_elements);
            sum += aCoeffs[k_filter] * from_data[from_actual_idx_in_slice];
        }
        to_data_half[i] = sum;
    }
}

void wavelet_noise_2d::Upsample1D(const std::vector<double>& from_data_half_size, std::vector<double>& to_data_full_size, int n_elements_full_size) const {
    to_data_full_size.assign(n_elements_full_size, 0.0);
    for (int i = 0; i < n_elements_full_size; ++i) {
        double sum = 0.0;
        int k0 = i / 2;
        int k1 = (i + 1) / 2;
        if (i % 2 == 0) {
            sum += pCoeffs[2] * from_data_half_size[Mod(k0, n_elements_full_size / 2)];
            sum += pCoeffs[0] * from_data_half_size[Mod(k1, n_elements_full_size / 2)];
        } else {
            sum += pCoeffs[3] * from_data_half_size[Mod(k0, n_elements_full_size / 2)];
            sum += pCoeffs[1] * from_data_half_size[Mod(k1, n_elements_full_size / 2)];
        }
        to_data_full_size[i] = sum;
    }
}


// --- 2D Tile Generation ---
void wavelet_noise_2d::GenerateNoiseTile2D(int n, int olap) {
    if (n % 2 != 0) n++;
    tile_size = n;
    int sz = n * n;
    noise_tile_data_2d.assign(sz, 0.0);
    std::vector<double> temp_processed_rows(sz); // Stores result after processing rows
    std::vector<double> final_processed_cols(sz); // Stores result after processing columns (this will be R_downarrow_uparrow)


    std::mt19937 gen(random_int(0,100000));

    // Step 1: Fill with random numbers (this is R)
    for (int i = 0; i < sz; ++i) {
        noise_tile_data_2d[i] = gaussian_noise(gen);
    }

    // Temporary storage for 1D slices
    std::vector<double> slice_from(n);
    std::vector<double> slice_to_half; // Size will be n/2
    std::vector<double> slice_to_full; // Size will be n

    // Steps 2 and 3: Downsample and Upsample for each dimension
    // Process along X rows (horizontal)
    for (int iy = 0; iy < n; ++iy) {
        // Extract X-row
        for (int ix = 0; ix < n; ++ix) {
            slice_from[ix] = noise_tile_data_2d[iy * n + ix];
        }
        Downsample1D(slice_from, slice_to_half, n);
        Upsample1D(slice_to_half, slice_to_full, n);
        // Place back into temp_processed_rows
        for (int ix = 0; ix < n; ++ix) {
            temp_processed_rows[iy * n + ix] = slice_to_full[ix];
        }
    }

    // Process along Y columns (vertical), using temp_processed_rows as input
    for (int ix = 0; ix < n; ++ix) {
        // Extract Y-column from temp_processed_rows
        for (int iy = 0; iy < n; ++iy) {
            slice_from[iy] = temp_processed_rows[iy * n + ix];
        }
        Downsample1D(slice_from, slice_to_half, n);
        Upsample1D(slice_to_half, slice_to_full, n);
        // Place back into final_processed_cols (this is R_downarrow_uparrow)
        for (int iy = 0; iy < n; ++iy) {
            final_processed_cols[iy * n + ix] = slice_to_full[iy];
        }
    }

    // Step 4: Subtract out the coarse-scale contribution ( N = R - R_downarrow_uparrow )
    for (int i = 0; i < sz; ++i) {
        noise_tile_data_2d[i] -= final_processed_cols[i];
    }

    // Avoid even/odd variance difference
    std::vector<double> temp_odd_offset(sz);
    int offset = n / 2;
    if (offset % 2 == 0) offset++;

    for (int iy = 0; iy < n; ++iy) {
        for (int ix = 0; ix < n; ++ix) {
            int current_idx = iy * n + ix;
            int offset_idx = Mod(iy + offset, n) * n + Mod(ix + offset, n);
            temp_odd_offset[current_idx] = noise_tile_data_2d[offset_idx];
        }
    }
    for (int i = 0; i < sz; ++i) {
        noise_tile_data_2d[i] += temp_odd_offset[i];
    }
    // noise_tile_data_2d is now ready
}


// --- 2D Noise Evaluation ---
void wavelet_noise_2d::evaluate_quadratic_bspline_weights(double p_val, double weights[3]) const {
    // Same as 3D version
    weights[0] = p_val * p_val / 2.0;
    weights[2] = (1.0 - p_val) * (1.0 - p_val) / 2.0;
    weights[1] = 1.0 - weights[0] - weights[2];
}

double wavelet_noise_2d::WNoise2D(const vec2& p_scaled) const {
    double result = 0.0;
    int mid[2];
    double w_basis[2][3];

    for (int i = 0; i < 2; ++i) {
        mid[i] = static_cast<int>(ceil(p_scaled[i] - 0.5));
        double t_paper = mid[i] - (p_scaled[i] - 0.5);
        evaluate_quadratic_bspline_weights(t_paper, w_basis[i]);
    }

    // Loop over 3x3 neighborhood
    for (int fy = -1; fy <= 1; ++fy) {
        for (int fx = -1; fx <= 1; ++fx) {
            double weight = 1.0;
            int c[2];

            weight *= w_basis[0][fx + 1];
            c[0] = Mod(mid[0] + fx, tile_size);

            weight *= w_basis[1][fy + 1];
            c[1] = Mod(mid[1] + fy, tile_size);

            result += weight * noise_tile_data_2d[c[1] * tile_size + c[0]];
        }
    }
    return result;
}

double wavelet_noise_2d::WMultibandNoise2D(const vec2& p, int nbands, double persistence) const {
    vec2 current_p = p;
    double result = 0.0;
    double amplitude = 1.0;
    double total_weight_sq = 0.0;

    for (int b = 0; b < nbands; ++b) {
        double band_noise = WNoise2D(current_p);
        result += amplitude * band_noise;
        total_weight_sq += amplitude * amplitude;

        current_p *= 2.0;
        amplitude *= persistence;
    }

    // Variance normalization (using 2D noise variance from paper's Section 4.2: 0.265)
    // Note: The paper says "For quadratic B-splines, this average σN^2 is 0.265 for 2D noise..."
    // This assumes the input random numbers for the tile had variance 1. Our gaussian_noise() does.
    if (total_weight_sq > 1e-9) {
        double avg_single_band_variance_2d = 0.265;
        // The goal of the normalization in the paper is so that `M(x)` has a predictable variance.
        // If `σ_N^2` is the variance of a single band `N(x)`, and `M(x) = sum w_b N(2^b x)`,
        // then `Var(M(x)) = (sum w_b^2) * σ_N^2`.
        // The paper's code in Appendix 2 effectively scales `result` so its variance becomes `avg_single_band_variance`.
        // `result /= sqrt(total_weight_sq / avg_single_band_variance)`
        // Variance of (result / C) = Var(result) / C^2.
        // We want new_var = avg_single_band_variance.
        // current_var = total_weight_sq * avg_single_band_variance_of_underlying_WNoise2D.
        // (assuming WNoise2D itself already has variance avg_single_band_variance_2d)
        // Let's assume WNoise2D has variance `V_wnoise2d`. Then `Var(result_before_norm) = total_weight_sq * V_wnoise2d`.
        // If `V_wnoise2d` is `avg_single_band_variance_2d`, then `result /= sqrt(total_weight_sq)`.
        // The paper's formula in `WMultiBandNoise` appendix 2 is:
        // `result /= sqrt(variance / ((normal) ? 0.296 : 0.210));` where `variance` is `sum w[b]*w[b]`.
        // This makes the final noise have a variance of 0.296 or 0.210.
        // So for 2D, it should be `result /= sqrt(total_weight_sq / 0.265);`
        result /= std::sqrt(total_weight_sq / avg_single_band_variance_2d);
    }
    return result;
}


// Public fractal noise methods
double wavelet_noise_2d::noise_2d(const vec2& p) const {
    return WNoise2D(p); // Assumes p is already at desired scale
}

double wavelet_noise_2d::fractal_noise_2d(const vec2& p, int octaves, double persistence) const {
    return WMultibandNoise2D(p, octaves, persistence);
}