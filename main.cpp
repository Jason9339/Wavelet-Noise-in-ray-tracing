#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rtweekend.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <random>

#include "sphere.h"  
#include "hittable.h" 
#include "hittable_list.h"
#include "texture.h"
#include "material.h"
#include "quad.h"
#include "perlin.h"

using namespace std;

using point3 = vec3;

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

const int MAX_DEPTH = 10; // 最多遞迴深度

// 噪聲類型枚舉
enum NoiseType {
    PERLIN_NOISE,
    WAVELET_3D_NOISE
};

vec3 trace(const ray& r, const hittable_list& world, int step, int max_step) {
    if (step > max_step) {
        return vec3(0, 0, 0); // 背景色 or 黑色
    }

    hit_record rec_nearest;
    
    if (!world.hit(r, 0.001f, FLT_MAX, rec_nearest)) {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1, 1, 1) + t * vec3(0.40, 0.50, 1.00);
    }

    color attenuation;
    ray scattered;
    
    if (rec_nearest.mat->scatter(r, rec_nearest, attenuation, scattered)) {
        return attenuation * trace(scattered, world, step + 1, max_step);
    }

    return rec_nearest.mat->emitted(rec_nearest.u, rec_nearest.v, rec_nearest.p);
}

shared_ptr<texture> create_noise_texture(NoiseType type, double scale = 1.0, int octave = 4) {
    switch (type) {
        case PERLIN_NOISE:
            return make_shared<noise_texture>(scale, octave);
        case WAVELET_3D_NOISE:
            return make_shared<wavelet_texture>(scale, octave, true);
        default:
            return make_shared<noise_texture>(scale, octave);
    }
}

void print_noise_options() {
    cout << "\n=== Wavelet Noise in Ray Tracing ===" << endl;
    cout << "可用的噪聲類型:" << endl;
    cout << "0 - Perlin Noise" << endl;
    cout << "1 - Wavelet 3D Noise" << endl; 
    cout << "=================================" << endl;
}

int main() {
    // 顯示噪聲選項
    print_noise_options();
    
    // 用戶選擇噪聲類型
    int noise_choice = 1; // 預設使用 Wavelet 3D
    cout << "選擇噪聲類型 (0-1，預設為1): ";
    if (!(cin >> noise_choice)) {
        noise_choice = 1;
        cin.clear();
        cin.ignore(10000, '\n');
    }
    
    if (noise_choice < 0 || noise_choice > 1) {
        noise_choice = 1;
    }
    
    NoiseType selected_noise = static_cast<NoiseType>(noise_choice);
    
    // 選擇 octave 級別（對所有噪聲類型都有效）
    int octave_level = 4;
    cout << "選擇 Octave 級別 (3-5，預設為4): ";
    if (!(cin >> octave_level)) {
        octave_level = 4;
        cin.clear();
        cin.ignore(10000, '\n');
    }
    if (octave_level < 3 || octave_level > 5) {
        octave_level = 4;
    }
    
    string noise_name;
    switch (selected_noise) {
        case PERLIN_NOISE: noise_name = "Perlin_octave" + to_string(octave_level); break;
        case WAVELET_3D_NOISE: noise_name = "Wavelet3D_octave" + to_string(octave_level); break;
    }
    
    cout << "\n使用噪聲類型: " << noise_name << endl;
    
    int width = 1000;
    int height = 500;
    int samples_per_pixel = 100;

    // 建立 buffer 儲存 RGB
    vector<unsigned char> image(width * height * 3);

    vec3 lower_left_corner(-2, -1, -1);
    vec3 origin(0, 0, 1);

    vec3 horizontal(4, 0, 0);
    vec3 vertical(0, 2, 0);

    cout << "origin : " << origin.x() << " " << origin.y() << " " << origin.z() << endl;

    // 創建材質  
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));  
    auto material1 = make_shared<lambertian>(color(0.7, 0.3, 0.3));  
    auto material2 = make_shared<metal>(color(0.8, 0.8, 0.9), 0.0);  
    auto material3 = make_shared<dielectric>(1.5);  
    
    // 創建光源材質 - 使用 diffuse_light
    auto light_material = make_shared<diffuse_light>(color(4.0, 4.0, 4.0));
    
    // 創建選定的噪聲紋理材質
    auto selected_noise_tex = create_noise_texture(selected_noise, 1.0, octave_level);
    auto noise_material = make_shared<lambertian>(selected_noise_tex);
      
    // 創建球體（使用材質）  
    hittable_list world;  
    
    // 創建水平平面作為地面（替代大球）
    auto ground_quad = make_shared<quad>(
        point3(-10, -0.5, -10),  // 平面左下角
        vec3(20, 0, 0),          // u 向量（水平方向，長度20）
        vec3(0, 0, 20),          // v 向量（深度方向，長度20）
        noise_material           // 使用選定的噪聲材質
    );
    world.add(ground_quad);
    
    // 添加光源球體（參考您原本的光源位置 -10, 10, 0）
    world.add(make_shared<sphere>(vec3(-5, 5, 0), 0.8, light_material));
    
    // 添加噪聲紋理球體
    world.add(make_shared<sphere>(vec3(1, 0, -1.75), 0.5, noise_material));  

    // 確保輸出目錄存在
    system("mkdir -p result_raytracing");
    
    string output_ppm = "result_raytracing/raytrace_" + noise_name + ".ppm";
    string output_png = "result_raytracing/raytrace_" + noise_name + ".png";

    ofstream file(output_ppm);
    file << "P3\n" << width << " " << height << "\n255\n";
    cout << "Processing..." << endl;
    
    for (int j = height - 1; j >= 0; --j) {
        // 進度顯示
        if (j % 50 == 0) {
            cout << "進度: " << (height - 1 - j) * 100 / height << "%" << endl;
        }
        
        for (int i = 0; i < width; ++i) {
            vec3 color_sum(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                float rand_u = float(rand()) / RAND_MAX - 0.5f;  
                float rand_v = float(rand()) / RAND_MAX - 0.5f;  
                float u = float(i + 0.5f + rand_u) / width;  
                float v = float(j + 0.5f + rand_v) / height;

                ray r(origin, unit_vector(lower_left_corner + u * horizontal + v * vertical - origin));
                color_sum += trace(r, world, 0, MAX_DEPTH);
            }

            vec3 c = color_sum / float(samples_per_pixel);
            int r = static_cast<int>(255.99 * clamp(c.x(), 0.0f, 1.0f));
            int g = static_cast<int>(255.99 * clamp(c.y(), 0.0f, 1.0f));
            int b = static_cast<int>(255.99 * clamp(c.z(), 0.0f, 1.0f));
            file << r << " " << g << " " << b << "\n";

            int index = ((height - 1 - j) * width + i) * 3;
            image[index + 0] = r;
            image[index + 1] = g;
            image[index + 2] = b;
        }
    }
    
    stbi_write_png(output_png.c_str(), width, height, 3, image.data(), width * 3);
    
    cout << "\n=== 渲染完成 ===" << endl;
    cout << "輸出文件:" << endl;
    cout << "- PPM: " << output_ppm << endl;
    cout << "- PNG: " << output_png << endl;
    cout << "使用噪聲類型: " << noise_name << endl;

    return 0;
}
