#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <random>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"  
#include "hittable.h" 
#include "hittable_list.h"
#include "texture.h"
#include "material.h"

using namespace std;

// 类型别名
using color = vec3;
using point3 = vec3;

inline double random_value() {
    return random_double();
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

const int MAX_DEPTH = 10; // 最多遞迴深度



vec3 trace(const ray& r, const hittable_list& world, int step, int max_step) {
    if (step > max_step) {
        return vec3(0, 0, 0); // 背景色 or 黑色
    }

    hit_record rec_nearest;
    
    // 使用 hittable_list 的 hit 方法來找最近的交點
    if (!world.hit(r, 0.001f, FLT_MAX, rec_nearest)) {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1, 1, 1) + t * vec3(0.40, 0.50, 1.00);
    }

    // 使用材质的scatter方法来处理光线反射/折射
    color attenuation;
    ray scattered;
    
    if (rec_nearest.mat->scatter(r, rec_nearest, attenuation, scattered)) {
        return attenuation * trace(scattered, world, step + 1, max_step);
    }
    
    // 如果材质不散射，返回黑色（或发光材质的颜色）
    return rec_nearest.mat->emitted(rec_nearest.u, rec_nearest.v, rec_nearest.p);
}


int main() {
    int width = 1000;
    int height = 500;
    int samples_per_pixel = 100;

    // 建立 buffer 儲存 RGB
    vector<unsigned char> image(width * height * 3);

    vec3 lower_left_corner(-2, -1, -1);
    vec3 origin(0, 0, 1);

    vec3 horizontal(4, 0, 0);
    vec3 vertical(0, 2, 0);

	// vec3 window_norm = unit_vector(cross(horizontal, vertical));
	// vec3 origin = ((lower_left_corner + horizontal + vertical)/2) + window_norm*3;

	cout << "origin : " << origin.x() << " " << origin.y() << " " << origin.z() << endl ;

    // 創建材質  
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));  
    auto material1 = make_shared<lambertian>(color(0.7, 0.3, 0.3));  
    auto material2 = make_shared<metal>(color(0.8, 0.8, 0.9), 0.0);  
    auto material3 = make_shared<dielectric>(1.5);  
    
    // 創建光源材質 - 使用 diffuse_light
    auto light_material = make_shared<diffuse_light>(color(4.0, 4.0, 4.0));
    
    // 添加 noise texture 材質
    auto noise_tex = make_shared<noise_texture>(4.0);
    auto noise_material = make_shared<lambertian>(noise_tex);
      
    // 創建球體（使用材質）  
    hittable_list world;  
    // world.add(make_shared<sphere>(vec3(0, -100.5, -2), 100, ground_material));  
    world.add(make_shared<sphere>(vec3(0, -100.5, -2), 100, noise_material)); 
    // world.add(make_shared<sphere>(vec3(0, 0, -2), 0.5, material1));  
    // world.add(make_shared<sphere>(vec3(1, 0, -1.75), 0.5, material2));  
    // world.add(make_shared<sphere>(vec3(-1, 0, -2.25), 0.5, material3));
    
    // 添加光源球體（參考您原本的光源位置 -10, 10, 0）
    world.add(make_shared<sphere>(vec3(-5, 5, 0), 0.8, light_material));
    
    // 添加 noise texture 球體
    world.add(make_shared<sphere>(vec3(2, 0, -1.25), 0.5, noise_material));  

    // srand(1234);
    // for (int i = 0; i < 48; i++) {
    //     float xr = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    //     float zr = ((float)rand() / RAND_MAX) * 3.0f - 1.5f;
  
    //     auto random_material = make_shared<lambertian>(color(random_value(), random_value(), random_value()));  
    //     world.add(make_shared<sphere>(vec3(xr, -0.45, zr - 2), 0.05, random_material));  
    // }

    ofstream file("../raytrace.ppm");
    file << "P3\n" << width << " " << height << "\n255\n";
	cout << "Processing" << endl;
    for (int j = height - 1; j >= 0; --j) {
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
    stbi_write_png("../raytrace.png", width, height, 3, image.data(), width * 3);
	cout << "End" << endl;

    return 0;
}
