#ifndef NOISE_UTILS_H
#define NOISE_UTILS_H

#include "perlin.h"
#include "wavelet_noise.h"      // <--- 新增 For 3D Wavelet Noise
#include "wavelet_noise_2d.h"   // <--- 新增 For 2D Wavelet Noise
#include "vec3.h"
#include "vec2.h"               // <--- 新增 For 2D points/vectors
#include <iostream>
#include <vector>
#include <string>

using point3 = vec3;

// 生成 2D Perlin 噪聲圖（使用 fractal_noise_2d）
void generate_noise_2d_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const std::string& filename = "noise_2d_map.png"
);

// 從 3D fractal noise 切出平面（使用 fractal_noise_3d）
void generate_noise_3d_plane_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const point3& plane_point = point3(0, 0, 0),
    const vec3& plane_normal = vec3(0, 1, 0),
    const std::string& filename = "noise_3d_plane_map.png"
);

// --- Wavelet Noise Generators ---

// Generates an image using pure 2D Wavelet Noise
void generate_wavelet_noise_2d_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const std::string& filename = "wavelet_noise_2d_map.png",
    int octaves = 7,
    double persistence = 0.5,
    int tile_size = 32
);

// Generates an image using single-band 2D Wavelet Noise (WNoise2D only)
void generate_wavelet_noise_2d_single_band_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const std::string& filename = "wavelet_noise_2d_single_band_map.png",
    int tile_size = 32
);

// Generates an image from a 2D slice of 3D Wavelet Noise (simple slicing)
void generate_wavelet_noise_3d_slice_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const point3& slice_plane_origin_offset = point3(100.123, 0.456, 87.789), // Offset to sample different part of noise
    int slice_axis = 2, // 0 for X=const, 1 for Y=const, 2 for Z=const
    double slice_coord = 0.0, // Coordinate value for the constant axis
    const std::string& filename = "wavelet_noise_3d_slice_map.png",
    int octaves = 7,
    double persistence = 0.5,
    int tile_size = 32
);

// Generates an image using single-band 3D Wavelet Noise slice (WNoise3D only)
void generate_wavelet_noise_3d_single_band_slice_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const point3& slice_plane_origin_offset = point3(100.123, 0.456, 87.789),
    int slice_axis = 2,
    double slice_coord = 0.0,
    const std::string& filename = "wavelet_noise_3d_single_band_slice_map.png",
    int tile_size = 32
);

// Generates an image by projecting 3D Wavelet Noise onto a 2D plane
void generate_wavelet_noise_3d_projected_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const point3& plane_point = point3(0, 0, 0),
    const vec3& plane_normal = vec3(0, 0, 1), // Default to XY plane, normal along Z
    const point3& noise_offset = point3(100.123, 0.456, 87.789), // Offset for sampling 3D noise
    const std::string& filename = "wavelet_noise_3d_projected_map.png",
    int octaves = 7,
    double persistence = 0.5,
    int tile_size = 32
);

// Generates an image using single-band 3D Wavelet Noise projection (WNoise3D only)
void generate_wavelet_noise_3d_single_band_projected_image(
    int width = 512,
    int height = 512,
    double scale = 1.0,
    double world_size = 20.0,
    const point3& plane_point = point3(0, 0, 0),
    const vec3& plane_normal = vec3(0, 0, 1),
    const point3& noise_offset = point3(100.123, 0.456, 87.789),
    const std::string& filename = "wavelet_noise_3d_single_band_projected_map.png",
    int tile_size = 32
);

// +++ 新增函數聲明 +++
// Generates an image representing a single, non-repeating tile of 2D Wavelet Noise
void generate_wavelet_noise_2d_single_tile_image(
    int image_size,         // Width and height of the output image, should ideally match tile_size_for_generator
    double input_coord_scale, // Scale factor for the input coordinates to WNoise2D, controls "frequency"
    const std::string& filename,
    int tile_size_for_generator // The tile size used to initialize the wavelet_noise_2d generator
);
// +++ 新增函數聲明結束 +++


#endif
// --- END OF FILE noise_utils.h ---