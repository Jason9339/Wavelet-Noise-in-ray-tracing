#include "stb_image_write.h"
#include "noise_utils.h"
#include <algorithm>
#include <cmath>

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
