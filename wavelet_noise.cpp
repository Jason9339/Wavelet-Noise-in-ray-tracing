#include "wavelet_noise.h"
#include "rtweekend.h"
#include <iostream> // For debugging

// Initialize static const members
const std::vector<double> wavelet_noise::aCoeffs = {
    0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002172, 0.019120,
    -0.005040, -0.044412, 0.011655, 0.103311, -0.025936, -0.243780, 0.033979, 0.655340, // Center (ARAD-1)
    0.655340, 0.033979, -0.243780, -0.025936, 0.103311, 0.011655, -0.044412, -0.005040,
    0.019120, 0.002172, -0.008233, -0.000938, 0.003546, 0.000410, -0.001528, 0.000334
}; // 32 elements, ARAD = 16 means center is index 15 and 16. Correct for direct use.

const std::vector<double> wavelet_noise::pCoeffs = {0.25, 0.75, 0.75, 0.25}; // 4 elements

int wavelet_noise::Mod(int x, int n) {
    int m = x % n;
    return (m < 0) ? m + n : m;
}

double wavelet_noise::gaussian_noise(std::mt19937& gen) {
    // Using Box-Muller transform for Gaussian distribution
    // For a simpler placeholder, one could use: return random_double(-1.0, 1.0);
    // However, paper mentions Gaussian for variance control (Sec 4.2)
    static std::normal_distribution<> d(0,1); // mean 0, stddev 1
    return d(gen);
}


wavelet_noise::wavelet_noise(int tile_size_) {
    if (tile_size_ <= 0) tile_size_ = 32; // Default minimum
    if (tile_size_ % 2 != 0) tile_size_++; // Must be even for upsample/downsample symmetry
    tile_size = tile_size_;
    
    GenerateNoiseTile(tile_size, 0); // olap=0 for now
}

// --- Appendix 1: GenerateNoiseTile ---
// Helper: Downsample a 1D slice of data
void wavelet_noise::Downsample(const std::vector<double>& from_data, std::vector<double>& to_data, 
                               int n_elements, int stride, const std::vector<double>& current_tile_slice_start) const {
    // 'from_data' provides access to the specific 1D slice via 'current_tile_slice_start' and 'stride'
    // 'to_data' is where the downsampled result for this slice is stored.
    // 'n_elements' is the number of elements in this 1D slice.
    // The paper's aCoeffs are already centered for direct use (ARAD is radius).
    // The 'from' in the paper's C code is the full 3D tile, and 'stride' selects row/col/depth.
    // Here, 'from_data' *is* the full 3D tile. 'current_tile_slice_start' points to the start of the 1D slice.
    
    // int to_idx = 0; // Index into to_data (which is just for this slice's result)
    for (int i = 0; i < n_elements / 2; ++i) {
        double sum = 0.0;
        for (int k_filter = 0; k_filter < 2 * ARAD; ++k_filter) { // Iterate through filter coeffs
            // k_abs is the absolute index in the 'from_data' for the current filter tap
            // 2*i is the center of the downsample operation in the 'from_data' space
            // k_filter - ARAD gives the relative offset from the center
            int from_relative_idx = 2 * i + (k_filter - ARAD);
            int from_actual_idx_in_slice = Mod(from_relative_idx, n_elements);
            
            // Accessing the 1D slice within the 3D 'from_data'
            // The 'from_data' vector is indexed as a flat array.
            // 'current_tile_slice_start' is the base pointer for the slice.
            // 'from_actual_idx_in_slice * stride' gives the offset from this base.
            // This reconstruction of indexing is tricky. Let's assume the caller passes a 1D slice directly.
            // For GenerateNoiseTile, we will extract 1D slices.

            // For this function to be generic, 'from_data' should be the 1D slice itself.
            // The caller (GenerateNoiseTile) will be responsible for extracting/placing 1D data.
            sum += aCoeffs[k_filter] * from_data[from_actual_idx_in_slice]; 
        }
        to_data[i] = sum; // Store in the (smaller) 'to_data' for this slice
    }
}


// Helper: Upsample a 1D slice of data
void wavelet_noise::Upsample(const std::vector<double>& from_data_half_size, std::vector<double>& to_data_full_size, 
                             int n_elements_full_size, int stride, const std::vector<double>& current_tile_slice_start) const {
    // 'from_data_half_size' is the downsampled 1D slice.
    // 'to_data_full_size' is where the upsampled result for this slice is stored.
    // 'n_elements_full_size' is the original number of elements in this 1D slice (before downsampling).
    // The paper's pCoeffs: p[-2]=0.25, p[-1]=0.75, p[0]=0.75, p[1]=0.25 relative to output sample.
    // Or, if pCoeffs[0..3] = {0.25, 0.75, 0.75, 0.25}
    // For output sample to[i*stride]:
    //  k=i/2 (integer div for from_idx)
    //  p_idx depends on whether i is even or odd relative to 2*k
    //  Let's use the paper's loop: for (int k=i/2; k<=i/2+1; k++) to[i*stride] += p[i-2*k] * from[Mod(k,n/2)*stride];
    // This means p is indexed relative to the output sample, centered at filter index 2 (or 1.5 effectively).
    // pCoeffs are {p0, p1, p2, p3}. Paper p is &pCoeffs[2]. So p[i-2k] accesses pCoeffs[2 + i - 2k].
    // i-2k can be:
    // if i is even, i=2m. k=m or m+1.
    //   i-2k = 0 (for k=m) -> pCoeffs[2]
    //   i-2k = -2 (for k=m+1) -> pCoeffs[0]
    // if i is odd, i=2m+1. k=m or m+1.
    //   i-2k = 1 (for k=m) -> pCoeffs[3]
    //   i-2k = -1 (for k=m+1) -> pCoeffs[1]
    
    // Corrected loop based on paper's Upsample:
    // for (int i=0; i<n; i++) { // n is n_elements_full_size
    //   to[i*stride] = 0;
    //   for (int k_spline=i/2; k_spline<=i/2+1; k_spline++) // Iterate over 2 input samples that contribute
    //     to[i*stride] += pCoeffs_centered[ i - 2*k_spline ] * from_data_half_size[Mod(k_spline, n_elements_full_size/2)];
    // }
    // Where pCoeffs_centered would be something like {..., pCoeffs[0] at index -2, pCoeffs[1] at -1, pCoeffs[2] at 0, pCoeffs[3] at 1, ...}
    // Or, using the given pCoeffs {0.25, 0.75, 0.75, 0.25} as p[0], p[1], p[2], p[3]:
    // The paper's `p = &pCoeffs[2]` and then `p[i-2*k]` means:
    //   if i-2k = 0, access pCoeffs[2] (0.75)
    //   if i-2k = 1, access pCoeffs[3] (0.25)
    //   if i-2k = -1, access pCoeffs[1] (0.75)
    //   if i-2k = -2, access pCoeffs[0] (0.25)

    for (int i = 0; i < n_elements_full_size; ++i) {
        double sum = 0.0;
        // The paper uses k=i/2 and k=i/2+1 with integer division, which covers two contributing samples from 'from_data_half_size'
        // For pCoeffs = {P0, P1, P2, P3}
        // If i is even, i=2m:
        //   k=m: i-2k = 0. Use P2. from_idx = m.
        //   k=m+1: i-2k = -2. Use P0. from_idx = m+1.
        // If i is odd, i=2m+1:
        //   k=m: i-2k = 1. Use P3. from_idx = m.
        //   k=m+1: i-2k = -1. Use P1. from_idx = m+1.
        
        int k0 = i / 2; // integer division
        int k1 = (i + 1) / 2; // effectively k0 or k0+1 depending on even/odd for symmetry with filter

        if (i % 2 == 0) { // Even output index
            sum += pCoeffs[2] * from_data_half_size[Mod(k0, n_elements_full_size / 2)]; // P2 for k=i/2
            sum += pCoeffs[0] * from_data_half_size[Mod(k1, n_elements_full_size / 2)]; // P0 for k=(i/2)+1 (approx)
        } else { // Odd output index
            sum += pCoeffs[3] * from_data_half_size[Mod(k0, n_elements_full_size / 2)]; // P3 for k=i/2
            sum += pCoeffs[1] * from_data_half_size[Mod(k1, n_elements_full_size / 2)]; // P1 for k=(i/2)+1 (approx)
        }
        to_data_full_size[i] = sum;
    }
}


void wavelet_noise::GenerateNoiseTile(int n, int olap) {
    if (n % 2 != 0) {
        // std::cerr << "Warning: Tile size must be even for Wavelet Noise. Adjusted." << std::endl;
        n++; // Ensure n is even
    }
    tile_size = n; // Update member if changed
    int sz = n * n * n;
    noise_tile_data.assign(sz, 0.0);
    std::vector<double> temp1(sz);
    std::vector<double> temp2(sz);

    std::mt19937 gen(random_int(0,100000)); // Seed with a random int

    // Step 1: Fill the tile with random numbers
    for (int i = 0; i < sz; ++i) {
        noise_tile_data[i] = gaussian_noise(gen);
    }

    // Steps 2 and 3: Downsample and Upsample for each dimension
    // This requires careful handling of 1D slices from the 3D data.

    // Process along X rows
    std::vector<double> slice_from(n);
    std::vector<double> slice_to_half(n / 2);
    std::vector<double> slice_to_full(n);

    for (int iy = 0; iy < n; ++iy) {
        for (int iz = 0; iz < n; ++iz) {
            // Extract X-row
            for (int ix = 0; ix < n; ++ix) {
                slice_from[ix] = noise_tile_data[iz * n * n + iy * n + ix];
            }
            // Downsample(from, to, n_elements, stride, data_ptr_start_of_slice)
            // The C code passes full array and stride. Here, pass extracted slice.
            Downsample(slice_from, slice_to_half, n, 1, slice_from /*not used this way*/); 
            Upsample(slice_to_half, slice_to_full, n, 1, slice_to_half /*not used*/);
            // Place back into temp2 (as per paper's logic)
            for (int ix = 0; ix < n; ++ix) {
                temp2[iz * n * n + iy * n + ix] = slice_to_full[ix];
            }
        }
    }
    
    // Process along Y rows (using temp2 as input, store in temp1 as intermediate for clarity)
    for (int ix = 0; ix < n; ++ix) {
        for (int iz = 0; iz < n; ++iz) {
            // Extract Y-column from temp2
            for (int iy = 0; iy < n; ++iy) {
                slice_from[iy] = temp2[iz * n * n + iy * n + ix];
            }
            Downsample(slice_from, slice_to_half, n, 1, slice_from);
            Upsample(slice_to_half, slice_to_full, n, 1, slice_to_half);
            // Place back into temp1
            for (int iy = 0; iy < n; ++iy) {
                temp1[iz * n * n + iy * n + ix] = slice_to_full[iy];
            }
        }
    }
    temp2 = temp1; // temp2 now holds X then Y processed data

    // Process along Z rows (using temp2 as input, store back in temp2)
    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            // Extract Z-column from temp2
            for (int iz = 0; iz < n; ++iz) {
                slice_from[iz] = temp2[iz * n * n + iy * n + ix];
            }
            Downsample(slice_from, slice_to_half, n, 1, slice_from);
            Upsample(slice_to_half, slice_to_full, n, 1, slice_to_half);
            // Place back into temp2
            for (int iz = 0; iz < n; ++iz) {
                temp2[iz * n * n + iy * n + ix] = slice_to_full[iz];
            }
        }
    }

    // Step 4: Subtract out the coarse-scale contribution
    for (int i = 0; i < sz; ++i) {
        noise_tile_data[i] -= temp2[i];
    }

    // Avoid even/odd variance difference by adding odd-offset version of noise to itself.
    // temp1 will store the odd-offset version.
    int offset = n / 2;
    if (offset % 2 == 0) offset++; // Ensure offset is odd

    for (int ix = 0; ix < n; ++ix) {
        for (int iy = 0; iy < n; ++iy) {
            for (int iz = 0; iz < n; ++iz) {
                int current_idx = iz * n * n + iy * n + ix;
                int offset_idx = Mod(iz + offset, n) * n * n + 
                                 Mod(iy + offset, n) * n + 
                                 Mod(ix + offset, n);
                temp1[current_idx] = noise_tile_data[offset_idx];
            }
        }
    }
    for (int i = 0; i < sz; ++i) {
        noise_tile_data[i] += temp1[i];
    }
    // noiseTileData is now ready
}


// --- Appendix 2: Noise Evaluation ---

void wavelet_noise::evaluate_quadratic_bspline_weights(double p_val, double weights[3]) const {
    // p_val is the fractional coordinate relative to the center of the evaluation cell.
    // B-spline basis functions for t-1, t, t+1 (relative to integer grid point)
    // Let point be P. Integer part is I. Fractional part is f = P-I.
    // We need weights for coefficients at I-1, I, I+1.
    // Or, if mid = ceil(P-0.5), then t = mid - (P-0.5) as in paper.
    // w[0] is for f-1, w[1] for f, w[2] for f+1 (if f is relative to cell center, e.g., p[i]-mid[i])
    // Paper: t = mid[i]-(p[i]-0.5); (p[i] is world coord, mid[i] is cell center)
    //    w[i][0]=t*t/2; w[i][2]=(1-t)*(1-t)/2; w[i][1]=1-w[i][0]-w[i][2];
    // This 't' is distance from P to (mid-0.5), which is the *left* edge of the cell centered at mid.
    // So t=0 at left edge, t=1 at right edge.
    // If p_val is this 't':
    weights[0] = (1.0 - p_val) * (1.0 - p_val) / 2.0;         // Weight for coefficient at right (index +1 from current cell start)
    weights[1] = 0.5 + p_val * (1.0 - p_val);                 // Weight for coefficient at middle (index +0 from current cell start)
                                                             // Or 1.0 - w[0] - w[2] if p_val is relative to some center
                                                             // (-(p_val-0.5)^2 + 3/4)
    weights[2] = p_val * p_val / 2.0;                         // Weight for coefficient at left (index -1 from current cell start)
    
    // Let's re-verify B-spline definition from Farin or standard texts.
    // For uniform quadratic B-spline N_{i,3}(u) with knots ...,i,i+1,i+2,i+3,...
    // If u is in [i+1, i+2): (local parameter t = u-(i+1))
    // N_{i,3}  (coeff at i)  : (1-t)^2 / 2
    // N_{i+1,3}(coeff at i+1): (-2t^2 + 2t + 1)/2  or (0.5 + t(1-t))
    // N_{i+2,3}(coeff at i+2): t^2 / 2
    // The paper's code `t=mid[i]-(p[i]-0.5); w[i][0]=t*t/2; w[i][2]=(1-t)*(1-t)/2; w[i][1]=1-w[i][0]-w[i][2];`
    // Here, `mid[i]` is the integer index of the central coefficient.
    // `p[i]` is the world coordinate.
    // `p[i]-0.5` shifts the coordinate system. `ceil(p[i]-0.5)` gives `mid[i]`.
    // So `t` is the fractional distance from `p[i]-0.5` to `mid[i]`.
    // Effectively, `t = (p[i]-0.5) - floor(p[i]-0.5)`. This is the standard fractional part.
    // If p_val is this standard fractional part `f`:
    // Coeff at floor(P) uses (1-f)^2/2
    // Coeff at floor(P)+1 uses (-2f^2+2f+1)/2
    // Coeff at floor(P)+2 uses f^2/2
    // The paper's `w[f[k]+1]` indexing implies its `w[0], w[1], w[2]` map to relative coefficient indices -1, 0, 1 from `mid[k]`.
    // So if `t` is `(p[k]-0.5) - floor(p[k]-0.5)`:
    // `w[k][0]` (for coeff at `mid[k]-1`) would be `t*t/2` from paper. This maps to `N_{i+2,3}` with `t` being `f`.
    // `w[k][1]` (for coeff at `mid[k]`) would be `1 - t*t/2 - (1-t)*(1-t)/2`. This maps to `N_{i+1,3}`.
    // `w[k][2]` (for coeff at `mid[k]+1`) would be `(1-t)*(1-t)/2`. This maps to `N_{i,3}`.
    // This means my weights array should be indexed [2], [1], [0] if `p_val` is `f`.
    // Or, if p_val is t from paper `t = mid[i]-(p[i]-0.5)`:
    //   w[0] (for mid-1) is (1-t)^2/2  -- No, paper has t*t/2
    //   w[1] (for mid) is 1-w[0]-w[2]
    //   w[2] (for mid+1) is t^2/2 -- No, paper has (1-t)*(1-t)/2
    // The paper's `w[k][f[k]+1]` where `f[k]` is -1, 0, 1 means:
    // `w[k][0]` -> coeff `mid[k]-1`
    // `w[k][1]` -> coeff `mid[k]`
    // `w[k][2]` -> coeff `mid[k]+1`

    // Let p_val be `frac_coord = p_world - floor(p_world)`
    // The coefficients involved are at floor(p_world)-1, floor(p_world), floor(p_world)+1
    // Or rather, the standard evaluation involves 3 basis functions centered at i-1, i, i+1 for interval [i, i+1]
    // using knots i-1, i, i+1, i+2.
    // N_{i-1,2}(x) = (1-(x-i))^2 / 2
    // N_{i,2}(x)   = ( (x-i)*(1+(x-i)) + (1-(x-i))*(x-i+1) )/2 = 0.5 + (x-i)(1-(x-i))
    // N_{i+1,2}(x) = (x-i)^2 / 2
    // Let `t_local = p_val`. (fractional part of p_world).
    // weights[0] is for coefficient at `floor(p_world)` (corresponds to paper's `mid[k]-1` if `mid[k]=ceil(p-0.5)` and `f[k]=-1`)
    // weights[1] is for coefficient at `floor(p_world)+1` (corresponds to paper's `mid[k]` if `f[k]=0`)
    // weights[2] is for coefficient at `floor(p_world)+2` (corresponds to paper's `mid[k]+1` if `f[k]=1`)
    // This mapping seems consistent with standard B-spline evaluation.
    // However, the paper's `t=mid[i]-(p[i]-0.5)` is actually `1.0 - frac_coord` if `p[i]-0.5` is used as base.
    // Let's use the paper's `t` directly for `w[0],w[1],w[2]` and be careful with `mid` and `f[k]`.

    // If p_val is `t_paper = mid_coord - (world_coord - 0.5)`:
    // weights[0] = for coeff at mid_coord - 1
    // weights[1] = for coeff at mid_coord
    // weights[2] = for coeff at mid_coord + 1
    // According to paper:
    // weights_paper[0] = t_paper * t_paper / 2.0;
    // weights_paper[2] = (1.0 - t_paper) * (1.0 - t_paper) / 2.0;
    // weights_paper[1] = 1.0 - weights_paper[0] - weights_paper[2];
    // These are assigned to `w[axis][0]`, `w[axis][2]`, `w[axis][1]` in the paper.
    // And then used as `w[i][f[i]+1]`.
    // So, if `f[i] = -1`, it uses `w[i][0]`. If `f[i]=0`, it uses `w[i][1]`. If `f[i]=1`, it uses `w[i][2]`.
    // This matches our interpretation:
    weights[0] = p_val * p_val / 2.0;                         // for coefficient at mid - 1
    weights[2] = (1.0 - p_val) * (1.0 - p_val) / 2.0;         // for coefficient at mid + 1
    weights[1] = 1.0 - weights[0] - weights[2];              // for coefficient at mid
}


double wavelet_noise::WNoise(const point3& p_scaled) const {
    // p_scaled is already frequency-scaled input point
    double result = 0.0;
    int mid[3];      // Integer part of cell center for p_scaled
    double w_basis[3][3]; // Basis function weights w[axis][coeff_idx_relative_to_mid]

    for (int i = 0; i < 3; ++i) {
        mid[i] = static_cast<int>(ceil(p_scaled[i] - 0.5));
        double t_paper = mid[i] - (p_scaled[i] - 0.5); // As in paper's Appendix 2
        evaluate_quadratic_bspline_weights(t_paper, w_basis[i]);
    }

    // Loop over 3x3x3 neighborhood of coefficients
    for (int fz = -1; fz <= 1; ++fz) {
        for (int fy = -1; fy <= 1; ++fy) {
            for (int fx = -1; fx <= 1; ++fx) {
                double weight = 1.0;
                int c[3]; // Actual noise coefficient indices

                weight *= w_basis[0][fx + 1]; // fx+1 maps -1,0,1 to 0,1,2 for w_basis index
                c[0] = Mod(mid[0] + fx, tile_size);

                weight *= w_basis[1][fy + 1];
                c[1] = Mod(mid[1] + fy, tile_size);

                weight *= w_basis[2][fz + 1];
                c[2] = Mod(mid[2] + fz, tile_size);
                
                result += weight * noise_tile_data[c[2] * tile_size * tile_size + c[1] * tile_size + c[0]];
            }
        }
    }
    return result;
}

double wavelet_noise::WProjectedNoise(const point3& p_scaled, const vec3& normal_world) const {
    // p_scaled is frequency-scaled. Normal is not frequency scaled (it's a direction).
    double result = 0.0;
    int c[3];         // Current coefficient location
    int min_idx[3], max_idx[3]; // Loop bounds for coefficients

    // Normalize normal, though it should be already for typical use
    vec3 normal = unit_vector(normal_world);

    // Bound the support of the basis functions for this projection direction
    // Paper: support = 3*abs(normal[i]) + 3*sqrt((1-normal[i]*normal[i])/2);
    // The 3*abs(normal[i]) is for the main axis projection.
    // The 3*sqrt((1-normal[i]*normal[i])/2) is for the extent in orthogonal plane.
    // This defines a bounding box around p_scaled for relevant coefficients.
    // The "3" comes from quadratic B-spline support width. For projected, it's wider.
    // Section 4.1 "double-width quadratic B-spline, whose support covers 6 coefficients"
    // Appendix 2 WProjectedNoise uses calculation that yields ~4.5 for orthogonal normal component.
    // `support = 3*abs(normal[i]) + 3*sqrt((1-normal[i]*normal[i])/2);` looks like support of 3 + 3/sqrt(2) ~ 3 + 2.12 = 5.12
    // If normal[i]=1, support=3. If normal[i]=0, support = 3/sqrt(2) ~ 2.12. This seems too small.
    // The quintic (Fig 7) has support 9. Approx by B(z/(2^(3/2))) implies width 3*2*sqrt(2) = 6*1.414 ~ 8.5 -- closer to 9.
    // Approximation by double-width B-spline has support 6. Let's use 6 for this calculation.
    // The formula in Appendix 2 for WProjectedNoise for `support` seems to be for the *half-width* of a B-spline for that axis.
    // The loop bounds min/max are `ceil(p[i] - support)` and `floor(p[i] + support)`.
    // Let's use their support calculation directly.
    for (int i = 0; i < 3; ++i) {
        double axis_support = 3.0 * std::abs(normal[i]) + 3.0 * std::sqrt((1.0 - normal[i] * normal[i]) / 2.0);
        min_idx[i] = static_cast<int>(std::ceil(p_scaled[i] - axis_support)); // Paper uses ceil(p - support)
        max_idx[i] = static_cast<int>(std::floor(p_scaled[i] + axis_support)); // Paper uses floor(p + support)
    }

    // Loop over the noise coefficients within the bound.
    for (c[2] = min_idx[2]; c[2] <= max_idx[2]; ++c[2]) {
        for (c[1] = min_idx[1]; c[1] <= max_idx[1]; ++c[1]) {
            for (c[0] = min_idx[0]; c[0] <= max_idx[0]; ++c[0]) {
                double dot_prod = 0.0;
                // Dot the normal with the vector from c to p_scaled
                for (int i = 0; i < 3; ++i) {
                    dot_prod += normal[i] * (p_scaled[i] - c[i]);
                }

                double combined_basis_weight = 1.0;
                // Evaluate the basis function at c moved halfway to p_scaled along the normal.
                // Paper: t = (c[i]+normal [i]*dot/2)-(p[i]-1.5);
                // This 't' is input to a B-spline evaluation.
                // B-spline: (t<=0||t>=3)? 0: (t<1) ? t*t/2: (t<2)? 1-( (t-1)*(t-1) + (2-t)*(2-t) )/2 : (3-t)*(3-t)/2;
                // This is a standard quadratic B-spline N(t) defined over [0,3].
                for (int i = 0; i < 3; ++i) {
                    double t_basis = (c[i] + normal[i] * dot_prod / 2.0) - (p_scaled[i] - 1.5);
                    double basis_val;
                    if (t_basis <= 0.0 || t_basis >= 3.0) {
                        basis_val = 0.0;
                    } else if (t_basis < 1.0) {
                        basis_val = t_basis * t_basis / 2.0;
                    } else if (t_basis < 2.0) {
                        double t1 = t_basis - 1.0;
                        double t2 = 2.0 - t_basis;
                        basis_val = 1.0 - (t1 * t1 + t2 * t2) / 2.0;
                    } else { // t_basis < 3.0
                        double t3 = 3.0 - t_basis;
                        basis_val = t3 * t3 / 2.0;
                    }
                    combined_basis_weight *= basis_val;
                }
                
                if (combined_basis_weight > 1e-9) { // Avoid adding tiny numbers if basis_val is effectively zero
                     result += combined_basis_weight * noise_tile_data[
                                Mod(c[2], tile_size) * tile_size * tile_size +
                                Mod(c[1], tile_size) * tile_size +
                                Mod(c[0], tile_size)];
                }
            }
        }
    }
    return result;
}


double wavelet_noise::WMultibandNoise(const point3& p, const vec3* normal, int nbands, double persistence) const {
    point3 q; // Scaled point for current band
    double result = 0.0;
    double amplitude = 1.0;
    double total_weight_sq = 0.0;

    // Paper's WMultibandNoise example seems to assume firstBand is high frequency (e.g., 0 or 1)
    // and s+firstBand+b < 0 is a termination condition (s is screen scale factor).
    // For fractal noise, we usually iterate octaves, increasing frequency.
    // Let's assume nbands is "octaves".
    // The `pow(2, firstBand+b)` in paper scales p. If firstBand=0, it's `p*2^b`.

    point3 current_p = p;
    for (int b = 0; b < nbands; ++b) {
        double band_noise;
        if (normal) {
            band_noise = WProjectedNoise(current_p, *normal);
        } else {
            band_noise = WNoise(current_p);
        }
        result += amplitude * band_noise;
        total_weight_sq += amplitude * amplitude;
        
        current_p *= 2.0;       // Increase frequency for next octave
        amplitude *= persistence; // Decrease amplitude
    }

    // Variance normalization (as per Appendix 2, but slightly simplified for general fractal use)
    // The paper's normalization `result /= sqrt(variance / ((normal) ? 0.296 : 0.210));`
    // where `variance` is `total_weight_sq`. This scales the output to have a std dev
    // of `sqrt(0.296)` or `sqrt(0.210)`.
    // For typical fractal noise, we might just want to normalize by total amplitude or not at all.
    // Let's implement the paper's normalization to be faithful.
    if (total_weight_sq > 1e-9) {
        double avg_single_band_variance = (normal) ? 0.296 : 0.210; // From Sec 4.2
        // If avg_single_band_variance is for noise with stddev 1, and our gaussian_noise() is stddev 1,
        // then WNoise output should have roughly this variance.
        result /= std::sqrt(total_weight_sq / avg_single_band_variance);
    }
    
    return result;
}


// Public fractal noise methods
double wavelet_noise::noise_3d(const point3& p) const {
    // This could be interpreted as the highest frequency band of a fractal noise,
    // or just WNoise applied to p without further scaling.
    // Let's make it WNoise on p directly, assuming p is already at the desired scale.
    return WNoise(p);
}

double wavelet_noise::projected_noise_3d(const point3& p, const vec3& normal) const {
    // Single-band projected 3D noise
    vec3 n_unit = unit_vector(normal);
    return WProjectedNoise(p, n_unit);
}

double wavelet_noise::fractal_noise_3d(const point3& p, int octaves, double persistence) const {
    return WMultibandNoise(p, nullptr, octaves, persistence);
}

double wavelet_noise::projected_fractal_noise_3d(const point3& p, const vec3& normal, int octaves, double persistence) const {
    vec3 n_unit = unit_vector(normal);
    return WMultibandNoise(p, &n_unit, octaves, persistence);
}