#ifndef NOISE_UTILS_H
#define NOISE_UTILS_H

#include "perlin.h"
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

#endif