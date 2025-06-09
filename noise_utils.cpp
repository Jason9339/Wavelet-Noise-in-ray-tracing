#include "stb_image_write.h"
#include "noise_utils.h"
#include <algorithm>
#include <cmath>
#include <map> // For static generator instances

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// 生成 2D 噪聲圖（使用 fractal_noise_2d）
void generate_noise_2d_image(int width, int height, double scale, double world_size, const std::string& filename) {
    perlin noise;
    std::vector<unsigned char> noise_image(width * height * 3);
    double min_val = 1.0, max_val = 0.0;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double x = ((double(i) / width) - 0.5) * world_size;
            double y = ((double(j) / height) - 0.5) * world_size;
            double raw = noise.fractal_noise_2d((x + 100.123) * scale, (y + 87.789) * scale);
            double n = clamp(0.5 + 0.5 * raw, 0.0, 1.0);
            min_val = std::min(min_val, n);
            max_val = std::max(max_val, n);
            int gray = static_cast<int>(n * 255.0);
            int idx = (j * width + i) * 3;
            noise_image[idx + 0] = gray;
            noise_image[idx + 1] = gray;
            noise_image[idx + 2] = gray;
        }
    }

    std::cout << "2D 噪聲範圍: [" << min_val << ", " << max_val << "]\n";
    stbi_write_png(filename.c_str(), width, height, 3, noise_image.data(), width * 3);
}

// 從 3D 噪聲中切出一個平面圖像（使用 fractal_noise_3d）
void generate_noise_3d_plane_image(int width, int height, double scale, double world_size,
                                   const point3& plane_point, const vec3& plane_normal,
                                   const std::string& filename) {
    perlin noise;
    std::vector<unsigned char> noise_image(width * height * 3);
    double min_val = 1.0, max_val = 0.0;

    vec3 normal = unit_vector(plane_normal);
    vec3 temp = (std::abs(normal.x()) < 0.9) ? vec3(1, 0, 0) : vec3(0, 1, 0);
    vec3 u = unit_vector(cross(normal, temp));
    vec3 v = unit_vector(cross(normal, u));

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double s = ((double(i) / width) - 0.5) * world_size;
            double t = ((double(j) / height) - 0.5) * world_size;
            point3 world_point = plane_point + s * u + t * v;
            double raw = noise.fractal_noise_3d((world_point + vec3(100.123, 0.456, 87.789)) * scale);
            double n = clamp(0.5 + 0.5 * raw, 0.0, 1.0);
            min_val = std::min(min_val, n);
            max_val = std::max(max_val, n);
            int gray = static_cast<int>(n * 255.0);
            int idx = (j * width + i) * 3;
            noise_image[idx + 0] = gray;
            noise_image[idx + 1] = gray;
            noise_image[idx + 2] = gray;
        }
    }

    std::cout << "3D 切片噪聲範圍: [" << min_val << ", " << max_val << "]\n";
    stbi_write_png(filename.c_str(), width, height, 3, noise_image.data(), width * 3);
}

// Helper for static generator instances to avoid re-initialization
template<typename NoiseType>
NoiseType& get_noise_generator(int tile_size) {
    static std::map<int, NoiseType> generators;
    if (generators.find(tile_size) == generators.end()) {
        generators.emplace(tile_size, NoiseType(tile_size));
    }
    return generators.at(tile_size);
}

// --- Wavelet Noise Functions ---

// Generates an image using pure 2D Wavelet Noise
void generate_wavelet_noise_2d_image(int width, int height, double scale, double world_size,
                                   const std::string& filename, int octaves, double persistence, int tile_size) {
    wavelet_noise_2d& wn_gen = get_noise_generator<wavelet_noise_2d>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    double min_val_stat = 1e10, max_val_stat = -1e10; // For raw noise statistics

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;
            
            vec2 p_world(u_norm * world_size, v_norm * world_size);
            // Apply scale and an arbitrary offset to sample different parts of the noise tile space
            vec2 p_sample = (p_world + vec2(100.123, 87.789)) * scale;

            double raw_noise = wn_gen.fractal_noise_2d(p_sample, octaves, persistence);
            
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);

            // Normalize based on typical wavelet noise output characteristics
            // Wavelet noise (normalized by paper's method) has std dev ~sqrt(0.265) ~= 0.515
            // So roughly 99.7% of values are within +/- 3*0.515 = +/- 1.545
            // Map [-1.545, 1.545] to [0,1] roughly => (raw_noise / (2*1.545)) + 0.5
            double n_val = clamp(0.5 + raw_noise / 3.09, 0.0, 1.0); 
            
            int gray = static_cast<int>(n_val * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Wavelet 2D Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// Generates an image from a 2D slice of 3D Wavelet Noise (simple slicing)
void generate_wavelet_noise_3d_slice_image(int width, int height, double scale, double world_size,
                                   const point3& slice_plane_origin_offset,
                                   int slice_axis, double slice_coord,
                                   const std::string& filename, int octaves, double persistence, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;

            point3 p_sample_plane;
            if (slice_axis == 0) { // X = const, plane is YZ
                p_sample_plane = point3(slice_coord, u_norm * world_size, v_norm * world_size);
            } else if (slice_axis == 1) { // Y = const, plane is XZ
                p_sample_plane = point3(u_norm * world_size, slice_coord, v_norm * world_size);
            } else { // Z = const, plane is XY
                p_sample_plane = point3(u_norm * world_size, v_norm * world_size, slice_coord);
            }
            
            point3 p_sample = (p_sample_plane + slice_plane_origin_offset) * scale;
            double raw_noise = wn_gen_3d.fractal_noise_3d(p_sample, octaves, persistence);

            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);

            // Normalize based on typical 3D wavelet noise output characteristics
            // Std dev ~sqrt(0.210) ~= 0.458. Range ~ +/- 3*0.458 = +/- 1.374
            double n_val = clamp(0.5 + raw_noise / 2.748, 0.0, 1.0);

            int gray = static_cast<int>(n_val * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Wavelet 3D Sliced Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// Generates an image by projecting 3D Wavelet Noise onto a 2D plane
void generate_wavelet_noise_3d_projected_image(int width, int height, double scale, double world_size,
                                   const point3& plane_point, const vec3& plane_normal,
                                   const point3& noise_offset,
                                   const std::string& filename, int octaves, double persistence, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    vec3 normal_unit = unit_vector(plane_normal);
    vec3 temp_u_calc = (std::abs(normal_unit.x()) < 0.99) ? vec3(1, 0, 0) : vec3(0, 1, 0); // Ensure temp_u_calc is not parallel to normal_unit
     if (dot(normal_unit, temp_u_calc) > 0.99 || dot(normal_unit, temp_u_calc) < -0.99) { // If still parallel (e.g. normal_unit=(0,1,0))
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

            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);

            // Normalize for projected 3D wavelet noise
            // Std dev ~sqrt(0.296) ~= 0.544. Range ~ +/- 3*0.544 = +/- 1.632
            double n_val = clamp(0.5 + raw_noise / 3.264, 0.0, 1.0);
            
            int gray = static_cast<int>(n_val * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Wavelet 3D Projected Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}

// --- Single-Band Wavelet Noise Functions ---

// Generates an image using single-band 2D Wavelet Noise (WNoise2D only)
void generate_wavelet_noise_2d_single_band_image(int width, int height, double scale, double world_size,
                                   const std::string& filename, int tile_size) {
    wavelet_noise_2d& wn_gen = get_noise_generator<wavelet_noise_2d>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    double min_val_stat = 1e10, max_val_stat = -1e10; // For raw noise statistics

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;
            
            vec2 p_world(u_norm * world_size, v_norm * world_size);
            // Apply scale and an arbitrary offset to sample different parts of the noise tile space
            vec2 p_sample = (p_world + vec2(100.123, 87.789)) * scale;

            // Use single-band noise only (WNoise2D directly)
            double raw_noise = wn_gen.noise_2d(p_sample);
            
            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);

            // Normalize based on single-band wavelet noise characteristics
            // For single band, the paper suggests that the noise has variance ~0.265 for 2D
            // So roughly 99.7% of values are within +/- 3*sqrt(0.265) = +/- 3*0.515 = +/- 1.545
            double n_val = clamp(0.5 + raw_noise / 3.09, 0.0, 1.0); 
            
            int gray = static_cast<int>(n_val * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single-band Wavelet 2D Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// Generates an image using single-band 3D Wavelet Noise slice (WNoise3D only)
void generate_wavelet_noise_3d_single_band_slice_image(int width, int height, double scale, double world_size,
                                   const point3& slice_plane_origin_offset,
                                   int slice_axis, double slice_coord,
                                   const std::string& filename, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double u_norm = (double(i) / width) - 0.5;
            double v_norm = (double(j) / height) - 0.5;

            point3 p_sample_plane;
            if (slice_axis == 0) { // X = const, plane is YZ
                p_sample_plane = point3(slice_coord, u_norm * world_size, v_norm * world_size);
            } else if (slice_axis == 1) { // Y = const, plane is XZ
                p_sample_plane = point3(u_norm * world_size, slice_coord, v_norm * world_size);
            } else { // Z = const, plane is XY
                p_sample_plane = point3(u_norm * world_size, v_norm * world_size, slice_coord);
            }
            
            point3 p_sample = (p_sample_plane + slice_plane_origin_offset) * scale;
            
            // Use single-band 3D noise only (WNoise3D directly)
            double raw_noise = wn_gen_3d.noise_3d(p_sample);

            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);

            // Normalize based on single-band 3D wavelet noise characteristics
            // For single band 3D, variance ~0.210. Range ~ +/- 3*sqrt(0.210) = +/- 3*0.458 = +/- 1.374
            double n_val = clamp(0.5 + raw_noise / 2.748, 0.0, 1.0);

            int gray = static_cast<int>(n_val * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single-band Wavelet 3D Sliced Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}


// Generates an image using single-band 3D Wavelet Noise projection (WNoise3D only)
void generate_wavelet_noise_3d_single_band_projected_image(int width, int height, double scale, double world_size,
                                   const point3& plane_point, const vec3& plane_normal,
                                   const point3& noise_offset,
                                   const std::string& filename, int tile_size) {
    wavelet_noise& wn_gen_3d = get_noise_generator<wavelet_noise>(tile_size);
    std::vector<unsigned char> image_data(width * height * 3);
    double min_val_stat = 1e10, max_val_stat = -1e10;

    vec3 normal_unit = unit_vector(plane_normal);
    vec3 temp_u_calc = (std::abs(normal_unit.x()) < 0.99) ? vec3(1, 0, 0) : vec3(0, 1, 0); // Ensure temp_u_calc is not parallel to normal_unit
     if (dot(normal_unit, temp_u_calc) > 0.99 || dot(normal_unit, temp_u_calc) < -0.99) { // If still parallel (e.g. normal_unit=(0,1,0))
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
            
            // Use single-band projected 3D noise only
            double raw_noise = wn_gen_3d.projected_noise_3d(p_sample, normal_unit);

            min_val_stat = std::min(min_val_stat, raw_noise);
            max_val_stat = std::max(max_val_stat, raw_noise);

            // Normalize for single-band projected 3D wavelet noise
            // For single band projected 3D, variance ~0.296. Range ~ +/- 3*sqrt(0.296) = +/- 3*0.544 = +/- 1.632
            double n_val = clamp(0.5 + raw_noise / 3.264, 0.0, 1.0);
            
            int gray = static_cast<int>(n_val * 255.0);
            int idx = (j * width + i) * 3;
            image_data[idx + 0] = gray;
            image_data[idx + 1] = gray;
            image_data[idx + 2] = gray;
        }
    }
    std::cout << "Single-band Wavelet 3D Projected Noise raw range: [" << min_val_stat << ", " << max_val_stat << "]. Image: " << filename << "\n";
    stbi_write_png(filename.c_str(), width, height, 3, image_data.data(), width * 3);
}
