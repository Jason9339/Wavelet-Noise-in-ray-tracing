#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rtweekend.h"
#include "perlin.h"
#include "vec3.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

// 配置參數結構體
struct NoiseConfig {
    int width = 512;           // 圖像寬度
    int height = 512;          // 圖像高度
    double scale = 5.0;        // 噪聲縮放因子
    double z_slice = 0.0;      // 在Z軸上的切片位置
    bool use_turbulence = false; // 是否使用湍流噪聲
    bool normalize = true;     // 是否標準化到[0,1]
    string output_name = "perlin_noise"; // 輸出檔案名
};

// 將Perlin噪聲值轉換為0-255的灰度值
int noise_to_grayscale(double noise_val, double min_val, double max_val, bool normalize) {
    if (normalize && (max_val > min_val)) {
        // 標準化到[0,1]
        noise_val = (noise_val - min_val) / (max_val - min_val);
    } else {
        // 簡單映射：將[-1,1]映射到[0,1]
        noise_val = (noise_val + 1.0) * 0.5;
    }
    
    // 確保在[0,1]範圍內
    noise_val = max(0.0, min(1.0, noise_val));
    
    return static_cast<int>(noise_val * 255.0);
}

// 生成Perlin噪聲灰度圖
void generate_perlin_image(const NoiseConfig& config) {
    perlin noise_gen;
    
    cout << "正在生成 " << config.width << "x" << config.height << " 的Perlin噪聲圖像..." << endl;
    cout << "參數設置：" << endl;
    cout << "  縮放因子: " << config.scale << endl;
    cout << "  Z切片位置: " << config.z_slice << endl;
    cout << "  使用湍流: " << (config.use_turbulence ? "是" : "否") << endl;
    cout << "  標準化: " << (config.normalize ? "是" : "否") << endl;
    
    // 創建圖像緩衝區
    vector<unsigned char> image(config.width * config.height * 3);
    
    // 第一次遍歷：找到噪聲值的範圍
    double min_noise = 1000.0;
    double max_noise = -1000.0;
    
    if (config.normalize) {
        cout << "正在分析噪聲值範圍..." << endl;
        for (int j = 0; j < config.height; j++) {
            for (int i = 0; i < config.width; i++) {
                // 將像素座標映射到噪聲空間
                double x = (double(i) / config.width) * config.scale;
                double y = (double(j) / config.height) * config.scale;
                double z = config.z_slice;
                
                point3 sample_point(x, y, z);
                double noise_val;
                
                if (config.use_turbulence) {
                    noise_val = noise_gen.turbulence_noise(sample_point);
                } else {
                    noise_val = noise_gen.noise(sample_point);
                }
                
                min_noise = min(min_noise, noise_val);
                max_noise = max(max_noise, noise_val);
            }
        }
        
        cout << "噪聲值範圍: [" << min_noise << ", " << max_noise << "]" << endl;
    }
    
    // 第二次遍歷：生成圖像
    cout << "正在生成圖像..." << endl;
    for (int j = 0; j < config.height; j++) {
        for (int i = 0; i < config.width; i++) {
            // 將像素座標映射到噪聲空間
            double x = (double(i) / config.width) * config.scale;
            double y = (double(j) / config.height) * config.scale;
            double z = config.z_slice;
            
            point3 sample_point(x, y, z);
            double noise_val;
            
            if (config.use_turbulence) {
                noise_val = noise_gen.turbulence_noise(sample_point);
            } else {
                noise_val = noise_gen.noise(sample_point);
            }
            
            // 轉換為灰度值
            int gray = noise_to_grayscale(noise_val, min_noise, max_noise, config.normalize);
            
            // 設置RGB（灰度圖像，所以R=G=B）
            int index = (j * config.width + i) * 3;
            image[index + 0] = gray; // R
            image[index + 1] = gray; // G
            image[index + 2] = gray; // B
        }
        
        // 顯示進度
        if (j % (config.height / 10) == 0) {
            cout << "進度: " << (j * 100 / config.height) << "%" << endl;
        }
    }
    
    // 保存為PNG和PPM格式
    string png_filename = config.output_name + ".png";
    string ppm_filename = config.output_name + ".ppm";
    
    // 保存PNG
    if (stbi_write_png(png_filename.c_str(), config.width, config.height, 3, image.data(), config.width * 3)) {
        cout << "PNG圖像已保存為: " << png_filename << endl;
    } else {
        cout << "保存PNG失敗！" << endl;
    }
    
    // 保存PPM
    ofstream ppm_file(ppm_filename);
    if (ppm_file.is_open()) {
        ppm_file << "P3\n" << config.width << " " << config.height << "\n255\n";
        
        for (int j = 0; j < config.height; j++) {
            for (int i = 0; i < config.width; i++) {
                int index = (j * config.width + i) * 3;
                ppm_file << static_cast<int>(image[index + 0]) << " "
                        << static_cast<int>(image[index + 1]) << " "
                        << static_cast<int>(image[index + 2]) << "\n";
            }
        }
        ppm_file.close();
        cout << "PPM圖像已保存為: " << ppm_filename << endl;
    } else {
        cout << "保存PPM失敗！" << endl;
    }
}

// 生成多個不同Z切片的圖像
void generate_z_slices(const NoiseConfig& base_config, int num_slices, double z_start, double z_end) {
    cout << "正在生成 " << num_slices << " 個Z切片圖像..." << endl;
    
    for (int i = 0; i < num_slices; i++) {
        NoiseConfig config = base_config;
        config.z_slice = z_start + (z_end - z_start) * i / (num_slices - 1);
        config.output_name = base_config.output_name + "_z" + to_string(i);
        
        cout << "\n=== 生成切片 " << (i + 1) << "/" << num_slices 
             << " (Z=" << config.z_slice << ") ===" << endl;
        
        generate_perlin_image(config);
    }
}

int main() {
    cout << "=== Perlin噪聲灰度圖生成器 ===" << endl;
    
    NoiseConfig config;
    
    // 基礎Perlin噪聲
    cout << "\n1. 生成基礎Perlin噪聲..." << endl;
    config.output_name = "perlin_basic";
    config.use_turbulence = false;
    config.scale = 5.0;
    generate_perlin_image(config);
    
    // 湍流Perlin噪聲
    cout << "\n2. 生成湍流Perlin噪聲..." << endl;
    config.output_name = "perlin_turbulence";
    config.use_turbulence = true;
    config.scale = 3.0;
    generate_perlin_image(config);
    
    // 不同尺度的噪聲
    cout << "\n3. 生成不同尺度的噪聲..." << endl;
    vector<double> scales = {1.0, 2.0, 5.0, 10.0, 20.0};
    for (double scale : scales) {
        config.output_name = "perlin_scale_" + to_string(static_cast<int>(scale));
        config.use_turbulence = false;
        config.scale = scale;
        generate_perlin_image(config);
    }
    
    // 生成Z軸切片序列
    cout << "\n4. 生成Z軸切片序列..." << endl;
    config.output_name = "perlin_slice";
    config.use_turbulence = false;
    config.scale = 5.0;
    generate_z_slices(config, 5, 0.0, 10.0);
    
    // 高解析度湍流噪聲
    cout << "\n5. 生成高解析度湍流噪聲..." << endl;
    config.width = 1024;
    config.height = 1024;
    config.output_name = "perlin_hires_turbulence";
    config.use_turbulence = true;
    config.scale = 8.0;
    generate_perlin_image(config);
    
    cout << "\n=== 所有圖像生成完成！===" << endl;
    
    return 0;
} 