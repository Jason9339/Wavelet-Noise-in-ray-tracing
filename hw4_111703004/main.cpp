#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cfloat>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"  
#include "hittable.h" 

using namespace std;

const int MAX_DEPTH = 5; // 最多遞迴深度

// 光照參數
vec3 lightsource1(-10, 10, 0);
vec3 I1 = vec3(1, 1, 1);
vec3 lightsource2(2, 1, -1);
vec3 I2 = vec3(0, 0, 0);

// 檢查陰影光線是否被遮擋
bool is_occluded(const vec3& origin, const vec3& light_dir, const vector<sphere>& world, float max_distance) {
    ray shadow_ray(origin, light_dir);
    for (const auto& obj : world) {
        hit_record rec;
		// 若超出 max_distance 這段距離才打中物體，則該物體在光源後方，不影響這個光源照到的結果。
		// 因為是單位向量，時間與距離值相等， tmax = max_distence
        if (obj.hit(shadow_ray, 0.001f, max_distance, rec)) {
            return true; // 有遮蔽
        }
    }
    return false;
}

// bonus
// 使用 Schlick approximation 計算 Fresnel 反射係數
float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

// bonus
// 計算從某點出發，沿著光線方向前進時，陰影光線的總體衰減係數
// 如果遇到完全不透明物體則完全遮蔽（返回 0）
// 否則根據物體的透明度 w_t 與 Beer-Lambert 定律逐層疊加
vec3 shadow_attenuation(vec3 origin, const vec3& light_dir, const vector<sphere>& world, float max_distance) {
    ray shadow_ray(origin, light_dir);
    vec3 attenuation(1.0f, 1.0f, 1.0f); // 初始為無衰減（全光照）

    // 嘗試追蹤 shadow_ray 是否會穿過多個透明物體（如玻璃）
    for (const auto& obj : world) {
        hit_record rec;
        // hit() 檢查是否在 max_distance 之內有交點
        if (obj.hit(shadow_ray, 0.001f, max_distance, rec)) {
            // 若物體完全不透明，光線無法穿透，直接遮蔽
            if (obj.w_t <= 0.0f) {
                return vec3(0.0f, 0.0f, 0.0f); // 陰影完全遮蔽
            }

            // 使用 Beer-Lambert 定律計算在透明物體中的衰減程度
            // 定律: I = I0 * exp(-σ * d)
            // σ: 吸收係數、d: 距離
            float distance_factor = rec.t / max_distance;

            float absorption = 0.5f; // 可調整，控制吸收強度
            // 根據材質顏色決定哪個頻段被吸收（如紅色玻璃不吸紅光）
            vec3 color_absorption = vec3(1.0f, 1.0f, 1.0f) - obj.Kd;

            vec3 beer_lambert = vec3(
                exp(-absorption * color_absorption.x() * distance_factor),
                exp(-absorption * color_absorption.y() * distance_factor),
                exp(-absorption * color_absorption.z() * distance_factor)
            );

            // 乘上材質的透光率 w_t，得到該物體的總衰減係數
            vec3 obj_attenuation = obj.w_t * beer_lambert;

            // 累乘衰減係數（多層透明物體會疊加衰減）
            attenuation = attenuation * obj_attenuation;

            // 若光幾乎被完全遮蔽，提早結束（避免多餘計算）
            if (attenuation.squared_length() < 0.01f) {
                return vec3(0.0f, 0.0f, 0.0f);
            }

            // 更新光線起點與剩餘距離，模擬光線穿透該透明物體後繼續前進
            origin = rec.p + light_dir * 0.001f; // 稍微偏移避免自相交
            max_distance -= rec.t;
            shadow_ray = ray(origin, light_dir);
        }
    }

    return attenuation; // 回傳最終累積的衰減結果
}

// 計算某點的漫反射光照，包含陰影與距離衰減的影響
vec3 diffuse(const hit_record& rec, const vector<sphere>& world) {
    vec3 p = rec.p;            // 交點位置
    vec3 N = rec.normal;       // 法線
    vec3 Kd = rec.Kd;          // 漫反射材質色

    // 光線方向（單位向量）
    vec3 L1 = unit_vector(lightsource1 - p);
    vec3 L2 = unit_vector(lightsource2 - p);

    // 光源與交點的距離（之後用於距離衰減與 shadow_ray 最大長度）
    float dist1 = (lightsource1 - p).length();
    float dist2 = (lightsource2 - p).length();

    // 計算陰影光線的衰減（考慮透明物體與遮蔽）
    vec3 atten1 = shadow_attenuation(p, L1, world, dist1);
    vec3 atten2 = shadow_attenuation(p, L2, world, dist2);

    // 漫反射強度 = 法線與光線夾角的餘弦值（夾角越小，照射越強）
    float diff1 = max(dot(N, L1), 0.0f);
    float diff2 = max(dot(N, L2), 0.0f);

    // 距離衰減（模擬光源遠近造成的強度變化，常見模型為 quadratic attenuation）
    float distance_atten1 = 1.0f / (1.0f + 0.01f * dist1 + 0.001f * dist1 * dist1);
    float distance_atten2 = 1.0f / (1.0f + 0.01f * dist2 + 0.001f * dist2 * dist2);

    // 每個光源的光照量 = 光強 * 入射角 * 陰影衰減 * 距離衰減
    vec3 light1 = I1 * diff1 * atten1 * distance_atten1;
    vec3 light2 = I2 * diff2 * atten2 * distance_atten2;

    // 回傳總光照乘上材質色
    return (light1 + light2) * Kd;
}

vec3 reflect(const vec3& v, const vec3& n) {
    // 反射公式：r = v - 2*(v·n)*n
    return v - 2 * dot(v, n) * n;
}

vec3 refract(const vec3& v, const vec3& n, float eta) {
    float cos_theta = fmin(dot(-v, n), 1.0f);
	// 垂直於法線的向量
    vec3 r_out_perp = eta * (v + cos_theta * n);
	// 畢氏定理算出 r_out_parallel 的大小平方
    float k = 1.0f - r_out_perp.squared_length();
    if (k < 0.0f) return vec3(0, 0, 0); // 全反射
    // 平行於法線的向量
	// 大小 : sqrt(k)
	// 方向 : -n
	vec3 r_out_parallel = -sqrt(k) * n;
    return r_out_perp + r_out_parallel;
}


vec3 trace(const ray& r, const vector<sphere>& world, int step, int max_step) {
    if (step > max_step) {
        return vec3(0, 0, 0); // 背景色 or 黑色
    }

    hit_record rec_nearest;
    float closest_so_far = FLT_MAX;
    bool hit_anything = false;

    // 遍歷所有物體，找最近的交點
    for (const auto& obj : world) {
        hit_record rec_temp;
        if (obj.hit(r, 0.001f, closest_so_far, rec_temp)) {
            hit_anything = true;
            closest_so_far = rec_temp.t;
            rec_nearest = rec_temp;
            rec_nearest.w_r = obj.w_r;
            rec_nearest.w_t = obj.w_t;
        }
    }

    if (!hit_anything) {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1, 1, 1) + t * vec3(0.40, 0.50, 1.00);
    }

    // 法線與入射點
    vec3 q = rec_nearest.p;
    vec3 N = rec_nearest.normal;
    vec3 V = unit_vector(r.direction());

    // 本地光照
    vec3 local_color = diffuse(rec_nearest, world);

    // 計算反射光
    vec3 r_dir = reflect(V, N);
    ray reflected_ray(q, r_dir);
    vec3 reflected_color = trace(reflected_ray, world, step + 1, max_step);

	float eta_air = 1.0f;
	float eta_glass = 1.46f;

	float eta = rec_nearest.front_face ? (eta_air / eta_glass) : (eta_glass / eta_air);

    // 計算折射光（使用 Snell's Law）
    vec3 t_dir = refract(V, N, eta);
    vec3 transmitted_color(0, 0, 0);
    if (t_dir.squared_length() > 0) {
        ray transmitted_ray(q, t_dir);
        transmitted_color = trace(transmitted_ray, world, step + 1, max_step);
    }

    // return (1 - rec_nearest.w_t )*((1-rec_nearest.w_r) * local_color + rec_nearest.w_r * reflected_color) + rec_nearest.w_t * transmitted_color;
    
    // Schlick Fresnel 反射係數
    float cos_theta = fmin(dot(-V, N), 1.0f);
    float reflect_prob = schlick(cos_theta, eta_glass); // 或 eta 根據方向

    vec3 color = vec3(0, 0, 0);
    
    // 非折射材質
    if (rec_nearest.w_t == 0.0f) {
        color = (1 - rec_nearest.w_r) * local_color + rec_nearest.w_r * reflected_color;
    } else {
        // 混合折射光與 Fresnel 比例的反射光
        vec3 mix_reflect_transmit = reflect_prob * reflected_color + (1.0f - reflect_prob) * transmitted_color;
        color = (1.0f - rec_nearest.w_t) * local_color + rec_nearest.w_t * mix_reflect_transmit;
    }

    return color;
}


int main() {
    int width = 1000;
    int height = 500;
    int samples_per_pixel = 10;

    // 建立 buffer 儲存 RGB
    vector<unsigned char> image(width * height * 3);

    vec3 lower_left_corner(-2, -1, -1);
    vec3 origin(0, 0, 1);

    vec3 horizontal(4, 0, 0);
    vec3 vertical(0, 2, 0);

	// vec3 window_norm = unit_vector(cross(horizontal, vertical));
	// vec3 origin = ((lower_left_corner + horizontal + vertical)/2) + window_norm*3;

	cout << "origin : " << origin.x() << " " << origin.y() << " " << origin.z() << endl ;

    vector<sphere> hitable_list;
    hitable_list.push_back(sphere(vec3(0, -100.5, -2), 100));
    hitable_list.push_back(sphere(vec3(0, 0, -2), 0.5, 0.0f, 0.9f));
    hitable_list.push_back(sphere(vec3(1, 0, -1.75), 0.5f, 1.0f, 0.0f));
    hitable_list.push_back(sphere(vec3(-1, 0, -2.25), 0.5f, 0.0f, 0.0f));

    srand(1234);
    for (int i = 0; i < 48; i++) {
        float xr = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
        float zr = ((float)rand() / RAND_MAX) * 3.0f - 1.5f;

		float wr = ((float)rand() / RAND_MAX) * 0.8f;               // 反射係數 [0, 0.8]
    	float wt = ((float)rand() / RAND_MAX) * (1.0f - wr);        // 折射係數 [0, 1 - wr]
  
        hitable_list.push_back(sphere(vec3(xr, -0.45, zr - 2), 0.05, wr, wt));
    }

    ofstream file("../raytrace.ppm");
    file << "P3\n" << width << " " << height << "\n255\n";
	cout << "Processing" << endl;
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            vec3 color_sum(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                float rand_u = float(rand()) / RAND_MAX;
                float rand_v = float(rand()) / RAND_MAX;
                float u = float(i + rand_u) / width;
                float v = float(j + rand_v) / height;

                ray r(origin, unit_vector(lower_left_corner + u * horizontal + v * vertical - origin));
                color_sum += trace(r, hitable_list, 0, MAX_DEPTH);
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
