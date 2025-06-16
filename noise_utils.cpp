#include "stb_image_write.h"
#include "noise_utils.h"
#include <algorithm>
#include <cmath>
#include <map> // For static generator instances

inline double clamp(double x, double min_val, double max_val) { // Renamed min, max to avoid conflict
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// 生成 2D 噪聲圖（使用 fractal_noise_2d）
void generate_noise_2d_image(int width, int height, double scale, double world_size, const std::string& filename) {
    perlin noise;
    std::vector<unsigned char> noise_image(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10; // Renamed to avoid conflict

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double x = ((double(i) / width) - 0.5) * world_size;
            double y = ((double(j) / height) - 0.5) * world_size;
            double raw = noise.fractal_noise_2d((x + 100.123) * scale, (y + 87.789) * scale);
            raw_values[j * width + i] = raw;
            min_val_stat = std::min(min_val_stat, raw); // Use renamed var
            max_val_stat = std::max(max_val_stat, raw); // Use renamed var
        }
    }

    double range = max_val_stat - min_val_stat; // Use renamed var
    if (range == 0) range = 1.0; // Avoid divide by zero
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range; // Use renamed var
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0); // clamp result of normalization
            int idx = (j * width + i) * 3;
            noise_image[idx + 0] = gray;
            noise_image[idx + 1] = gray;
            noise_image[idx + 2] = gray;
        }
    }

    std::cout << "Perlin 2D Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n"; // Clarified type
    stbi_write_png(filename.c_str(), width, height, 3, noise_image.data(), width * 3);
}

// 從 3D 噪聲中切出一個平面圖像（使用 fractal_noise_3d）
void generate_noise_3d_plane_image(int width, int height, double scale, double world_size,
                                   const point3& plane_point, const vec3& plane_normal,
                                   const std::string& filename) {
    perlin noise;
    std::vector<unsigned char> noise_image(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10; // Renamed

    vec3 normal = unit_vector(plane_normal);
    vec3 temp_u_calc = (std::abs(normal.x()) < 0.99) ? vec3(1,0,0) : vec3(0,1,0);
    if (std::abs(dot(normal, temp_u_calc)) > 0.999) { // Check if normal is along y or x after first attempt
         temp_u_calc = vec3(0,0,1); // Use z-axis if normal is x or y
    }
    vec3 u = unit_vector(cross(normal, temp_u_calc));
    vec3 v = unit_vector(cross(normal, u));


    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double s = ((double(i) / width) - 0.5) * world_size;
            double t = ((double(j) / height) - 0.5) * world_size;
            point3 world_point = plane_point + s * u + t * v;
            double raw = noise.fractal_noise_3d((world_point + vec3(100.123, 0.456, 87.789)) * scale);
            raw_values[j * width + i] = raw;
            min_val_stat = std::min(min_val_stat, raw); // Use renamed
            max_val_stat = std::max(max_val_stat, raw); // Use renamed
        }
    }

    double range = max_val_stat - min_val_stat; // Use renamed
    if (range == 0) range = 1.0;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range; // Use renamed
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            noise_image[idx + 0] = gray;
            noise_image[idx + 1] = gray;
            noise_image[idx + 2] = gray;
        }
    }

    std::cout << "Perlin 3D Sliced Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n"; // Clarified
    stbi_write_png(filename.c_str(), width, height, 3, noise_image.data(), width * 3);
}

// Helper for static generator instances to avoid re-initialization
template<typename NoiseType>
NoiseType& get_noise_generator(int tile_size) {
    static std::map<int, NoiseType> generators;
    if (generators.find(tile_size) == generators.end()) {
        // std::cout << "Creating new " << typeid(NoiseType).name() << " generator with tile_size: " << tile_size << std::endl;
        generators.emplace(std::piecewise_construct,
                         std::forward_as_tuple(tile_size),
                         std::forward_as_tuple(tile_size));
    }
    return generators.at(tile_size);
}

// --- Wavelet Noise Functions ---

// Generates an image using pure 2D Wavelet Noise
void generate_wavelet_noise_2d_image(int width, int height, double scale, double world_size,
                                   const std::string& filename, int octaves, double persistence, int tile_size) {
    wavelet_noise_2d& wn_gen = get_noise_generator<wavelet_noise_2d>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;
            
            vec2 p_world(u_norm * world_size, v_norm * world_size);
            vec2 p_sample = (p_world + vec2(100.123, 87.789)) * scale;
            double raw_noise = wn_gen.fractal_noise_2d(p_sample, octaves, persistence);

            raw_values[j * width + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0; // More robust check for near-zero range
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Wavelet 2D Fractal Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// Generates an image from a 2D slice of 3D Wavelet Noise (simple slicing)
void generate_wavelet_noise_3d_slice_image(int width, int height, double scale, double world_size,
                                   const point3& slice_plane_origin_offset,
                                   int slice_axis, double slice_coord,
                                   const std::string& filename, int octaves, double persistence, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;

            point3 p_sample_plane;
            if (slice_axis == 0) { 
                p_sample_plane = point3(slice_coord, u_norm * world_size, v_norm * world_size);
            } else if (slice_axis == 1) { 
                p_sample_plane = point3(u_norm * world_size, slice_coord, v_norm * world_size);
            } else { 
                p_sample_plane = point3(u_norm * world_size, v_norm * world_size, slice_coord);
            }
            
            point3 p_sample = (p_sample_plane + slice_plane_origin_offset) * scale;
            double raw_noise = wn_gen_3d.fractal_noise_3d(p_sample, octaves, persistence);

            raw_values[j * width + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Wavelet 3D Sliced Fractal Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// Generates an image by projecting 3D Wavelet Noise onto a 2D plane
void generate_wavelet_noise_3d_projected_image(int width, int height, double scale, double world_size,
                                   const point3& plane_point, const vec3& plane_normal,
                                   const point3& noise_offset,
                                   const std::string& filename, int octaves, double persistence, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    vec3 normal_unit = unit_vector(plane_normal);
    vec3 temp_u_calc = (std::abs(normal_unit.x()) < 0.99) ? vec3(1,0,0) : vec3(0,1,0);
    if (std::abs(dot(normal_unit, temp_u_calc)) > 0.999) {
         temp_u_calc = vec3(0,0,1);
    }
    vec3 u_basis = unit_vector(cross(normal_unit, temp_u_calc));
    vec3 v_basis = unit_vector(cross(normal_unit, u_basis));

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double s_param = ((double(i) / width) - 0.5) * world_size;
            double t_param = ((double(j) / height) - 0.5) * world_size;
            
            point3 p_world_on_plane = plane_point + s_param * u_basis + t_param * v_basis;
            point3 p_sample = (p_world_on_plane + noise_offset) * scale;
            
            double raw_noise = wn_gen_3d.projected_fractal_noise_3d(p_sample, normal_unit, octaves, persistence);

            raw_values[j * width + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Wavelet 3D Projected Fractal Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}

// --- Single-Band Wavelet Noise Functions ---

void generate_wavelet_noise_2d_single_band_image(int width, int height, double scale, double world_size,
                                   const std::string& filename, int tile_size) {
    wavelet_noise_2d& wn_gen = get_noise_generator<wavelet_noise_2d>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;
            vec2 p_world(u_norm * world_size, v_norm * world_size);
            vec2 p_sample = (p_world + vec2(100.123, 87.789)) * scale;
            double raw_noise = wn_gen.noise_2d(p_sample);

            raw_values[j * width + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single-band Wavelet 2D Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}

void generate_wavelet_noise_3d_single_band_slice_image(int width, int height, double scale, double world_size,
                                   const point3& slice_plane_origin_offset,
                                   int slice_axis, double slice_coord,
                                   const std::string& filename, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;
            point3 p_sample_plane;
            if (slice_axis == 0) { 
                p_sample_plane = point3(slice_coord, u_norm * world_size, v_norm * world_size);
            } else if (slice_axis == 1) { 
                p_sample_plane = point3(u_norm * world_size, slice_coord, v_norm * world_size);
            } else { 
                p_sample_plane = point3(u_norm * world_size, v_norm * world_size, slice_coord);
            }
            point3 p_sample = (p_sample_plane + slice_plane_origin_offset) * scale;
            double raw_noise = wn_gen_3d.noise_3d(p_sample);

            raw_values[j * width + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single-band Wavelet 3D Sliced Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}

void generate_wavelet_noise_3d_single_band_projected_image(int width, int height, double scale, double world_size,
                                   const point3& plane_point, const vec3& plane_normal,
                                   const point3& noise_offset,
                                   const std::string& filename, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    std::vector<double> raw_values(width * height);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    vec3 normal_unit = unit_vector(plane_normal);
    vec3 temp_u_calc = (std::abs(normal_unit.x()) < 0.99) ? vec3(1,0,0) : vec3(0,1,0);
     if (std::abs(dot(normal_unit, temp_u_calc)) > 0.999) {
        temp_u_calc = vec3(0,0,1);
    }
    vec3 u_basis = unit_vector(cross(normal_unit, temp_u_calc));
    vec3 v_basis = unit_vector(cross(normal_unit, u_basis));

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double s_param = ((double(i) / width) - 0.5) * world_size;
            double t_param = ((double(j) / height) - 0.5) * world_size;
            point3 p_world_on_plane = plane_point + s_param * u_basis + t_param * v_basis;
            point3 p_sample = (p_world_on_plane + noise_offset) * scale;
            double raw_noise = wn_gen_3d.projected_noise_3d(p_sample, normal_unit);

            raw_values[j * width + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double n_val = (raw_values[j * width + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single-band Wavelet 3D Projected Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// +++ 新增函數實現 +++
// In noise_utils.cpp
void generate_wavelet_noise_2d_single_tile_image(
    int image_size,         // Width and height of the output image
    double sampling_density_scale, // Renamed from input_coord_scale.
                                  // 1.0 means one pixel in image maps to one unit in noise space for the base tile.
                                  // >1.0 means higher frequency sampling from the tile (zooming in on details).
                                  // <1.0 means lower frequency (zooming out, seeing larger features of the tile).
    const std::string& filename,
    int tile_size_for_generator) {

    if (image_size <= 0 || tile_size_for_generator <= 0) {
        std::cerr << "Error: image_size and tile_size_for_generator must be positive." << std::endl;
        return;
    }

    wavelet_noise_2d& wn_gen = get_noise_generator<wavelet_noise_2d>(tile_size_for_generator);

    std::vector<unsigned char> image_data(image_size * image_size * 3);
    std::vector<double> raw_values(image_size * image_size);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < image_size; ++j) {
        for (int i = 0; i < image_size; ++i) {
            // Map image pixel (i,j) to input coordinates for WNoise2D
            // We want the p_input to WNoise2D to effectively scan across
            // a single logical tile region, possibly scaled by sampling_density_scale.

            // norm_i, norm_j range from 0 to almost 1
            double norm_i = static_cast<double>(i) / image_size; 
            double norm_j = static_cast<double>(j) / image_size;

            // p_base represents coordinates that would span exactly one tile
            // if image_size == tile_size_for_generator.
            // More generally, it maps the image coordinates [0, image_size-1]
            // to noise coordinates [0, tile_size_for_generator-1].
            // This means each "unit" in noise coordinate space corresponds to
            // (image_size / tile_size_for_generator) pixels in the image.
            vec2 p_base(
                norm_i * tile_size_for_generator, 
                norm_j * tile_size_for_generator  
            );
            
            // sampling_density_scale adjusts how "densely" we sample from this base tile region.
            // If sampling_density_scale = 1.0, and image_size == tile_size_for_generator,
            // then p_input effectively goes from [0, tile_size_for_generator).
            // If sampling_density_scale = 2.0, we are effectively sampling twice as "fast"
            // from the same tile region, showing higher frequency details of that single tile.
            // The input to WNoise2D will be scaled. WNoise2D itself uses Mod(..., tile_size_for_generator).
            // So if p_base * sampling_density_scale exceeds tile_size_for_generator, tiling occurs.
            //
            // TO ACHIEVE A TRULY NON-REPEATING SINGLE TILE IMAGE:
            // The coordinates passed to wn_gen.noise_2d() MUST stay within the range
            // [0, tile_size_for_generator - epsilon) to avoid the Mod() causing repeats.
            //
            // Let's redefine:
            // image_size: size of the output image.
            // tile_size_for_generator: the tile size of the wn_gen.
            // We want to render an image of size `image_size` that shows content from
            // a single tile of `wn_gen`.
            // The `sampling_density_scale` will control how much of that tile's "detail" we see.

            vec2 p_input_for_wnoise;
            if (image_size == tile_size_for_generator) {
                // If image size matches generator tile size, map pixel 1-to-1 to noise coord units, then scale.
                // The effective coordinate range for WNoise2D will be [0, tile_size_for_generator * sampling_density_scale).
                // If sampling_density_scale > 1, this will cause tiling within WNoise2D due to Mod.
                // This is what happened in your problematic image.

                // To show a single, non-repeating tile, sampling_density_scale should effectively be 1.0
                // OR, the coordinates sent to WNoise2D must not exceed tile_size_for_generator.

                // Correct approach for non-repeating single tile image:
                // The input to WNoise2D should span [0, tile_size_for_generator - epsilon).
                // The `sampling_density_scale` now means: what is the frequency content *within* this single tile.
                // The name was perhaps misleading. Let's call it `frequency_multiplier_within_tile`.
                
                // Base coordinates that span [0, tile_size_for_generator - epsilon)
                // when image pixel goes from 0 to image_size-1.
                // This means each pixel in the image samples a point in the tile.
                p_input_for_wnoise.e[0] = (static_cast<double>(i) / (image_size - 1)) * (tile_size_for_generator - 1e-4);
                p_input_for_wnoise.e[1] = (static_cast<double>(j) / (image_size - 1)) * (tile_size_for_generator - 1e-4);
                // Now, `frequency_multiplier_within_tile` (was sampling_density_scale) scales these.
                p_input_for_wnoise *= sampling_density_scale; // Renamed for clarity

            } else {
                 // If image_size != tile_size_for_generator, the mapping is more complex
                 // to ensure we are looking at a "single tile's worth" of content without repeat.
                 // For simplicity, for this specific test, ensure image_size == tile_size_for_generator.
                 // Or, adjust p_input to always be within [0, tile_size_for_generator * sampling_density_scale)
                 // and accept that if sampling_density_scale > 1, it's showing multiple logical periods
                 // of the base band-limited noise signal, but still within one WNoise2D tile data structure.

                 // Let's stick to the interpretation that `sampling_density_scale` controls the
                 // frequency of the noise *within the displayed single tile image*.
                 // The image itself has `image_size` pixels.
                 // The coordinates sent to `WNoise2D` will be scaled by `sampling_density_scale`.
                 // `WNoise2D` uses a tile of `tile_size_for_generator`.
                 // If `(image_pixel_coord / image_size) * image_size * sampling_density_scale` (i.e. `image_pixel_coord * sampling_density_scale`)
                 // exceeds `tile_size_for_generator`, then `Mod` will kick in.

                 // The most straightforward way to get a non-repeating image of one tile:
                 // 1. Set image_size = tile_size_for_generator.
                 // 2. The coordinates passed to WNoise2D should be `vec2(i * scale_factor, j * scale_factor)`.
                 //    If scale_factor = 1.0, you see the tile 1:1.
                 //    If scale_factor > 1.0, you see higher frequencies from that tile (zoomed in on detail,
                 //    so the input coords to WNoise2D will exceed tile_size_for_generator causing tiling).
                 //    If scale_factor < 1.0, you see lower frequencies (zoomed out, effectively).

                 // Let's simplify the logic for the "single_tile_image" function:
                 // The image produced will be `image_size x image_size`.
                 // The `sampling_density_scale` (previously `input_coord_scale`) determines
                 // how many "periods" of the base noise band are packed into this image.
                 // The WNoise2D function will use its internal tile of `tile_size_for_generator`.
                 p_input_for_wnoise.e[0] = (static_cast<double>(i) / image_size) * image_size * sampling_density_scale;
                 p_input_for_wnoise.e[1] = (static_cast<double>(j) / image_size) * image_size * sampling_density_scale;
                 // Example: image_size=128, sampling_density_scale=1.0. p_input goes from 0 to 127. Correct for one tile period.
                 // Example: image_size=128, sampling_density_scale=4.0. p_input goes from 0 to 511. This will show 4x4 tiling
                 // from the WNoise2D generator whose tile_size is 128. This IS what your image shows.
            }


            double raw_noise = wn_gen.noise_2d(p_input_for_wnoise);

            raw_values[j * image_size + i] = raw_noise;
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);
        }
    }
    // ... (rest of the normalization and saving code remains the same) ...
    // (Make sure to use sampling_density_scale in the cout message)
    double range = max_val_stat - min_val_stat;
    if (range < 1e-9) range = 1.0; 
    for (int j = 0; j < image_size; ++j) {
        for (int i = 0; i < image_size; ++i) {
            double n_val = (raw_values[j * image_size + i] - min_val_stat) / range;
            int gray = static_cast<int>(clamp(n_val, 0.0, 1.0) * 255.0); 
            int idx = (j * image_size + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single Tile Test Wavelet 2D Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename 
              << " (ImageSize: " << image_size << ", SamplingDensityScale: " << sampling_density_scale // Updated name
              << ", GenTileSize: " << tile_size_for_generator << ")\n";
    stbi_write_png(filename.c_str(), image_size, image_size, 3, image_data.data(), image_size * 3);
}