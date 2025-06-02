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
#include <sstream>

using namespace std;

// 配置參數結構體
struct NoiseConfig {
    int width = 512;
    int height = 512;
    double scale = 5.0;
    double z_slice = 0.0;
    bool use_turbulence = false;
    bool normalize = true;
    string output_name = "perlin_noise";
    
    // 觀察平面參數
    char plane = 'z';  // 'x', 'y', 或 'z'
    double plane_position = 0.0;
};

// 獲取用戶輸入的函數
template<typename T>
T get_input(const string& prompt, T default_value) {
    string input;
    cout << prompt << " [預設: " << default_value << "]: ";
    getline(cin, input);
    
    if (input.empty()) {
        return default_value;
    }
    
    istringstream iss(input);
    T value;
    if (iss >> value) {
        return value;
    } else {
        cout << "輸入無效，使用預設值: " << default_value << endl;
        return default_value;
    }
}

// 特化為字串
template<>
string get_input<string>(const string& prompt, string default_value) {
    string input;
    cout << prompt << " [預設: " << default_value << "]: ";
    getline(cin, input);
    
    if (input.empty()) {
        return default_value;
    }
    return input;
}

// 特化為字元
template<>
char get_input<char>(const string& prompt, char default_value) {
    string input;
    cout << prompt << " [預設: " << default_value << "]: ";
    getline(cin, input);
    
    if (input.empty() || input.length() != 1) {
        return default_value;
    }
    return input[0];
}

// 獲取yes/no輸入
bool get_yes_no(const string& prompt, bool default_value) {
    string input;
    cout << prompt << " (y/n) [預設: " << (default_value ? "y" : "n") << "]: ";
    getline(cin, input);
    
    if (input.empty()) {
        return default_value;
    }
    
    return (input[0] == 'y' || input[0] == 'Y');
}

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

// 根據觀察平面獲取採樣點
point3 get_sample_point(int i, int j, const NoiseConfig& config) {
    double u = (double(i) / config.width) * config.scale;
    double v = (double(j) / config.height) * config.scale;
    
    switch (config.plane) {
        case 'x':
            return point3(config.plane_position, u, v);
        case 'y':
            return point3(u, config.plane_position, v);
        case 'z':
        default:
            return point3(u, v, config.plane_position);
    }
}

// 生成Perlin噪聲灰度圖
void generate_perlin_image(const NoiseConfig& config) {
    perlin noise_gen;
    
    cout << "\n正在生成 " << config.width << "x" << config.height << " 的Perlin噪聲圖像..." << endl;
    cout << "參數設置：" << endl;
    cout << "  縮放因子: " << config.scale << endl;
    cout << "  觀察平面: " << config.plane << " = " << config.plane_position << endl;
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
                point3 sample_point = get_sample_point(i, j, config);
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
            point3 sample_point = get_sample_point(i, j, config);
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
        if (j % max(1, config.height / 10) == 0) {
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

// 顯示選單
void show_menu() {
    cout << "\n=== Perlin噪聲灰度圖生成器 ===" << endl;
    cout << "1. 自訂參數生成" << endl;
    cout << "2. 快速預設生成" << endl;
    cout << "3. 生成動畫序列" << endl;
    cout << "4. 退出" << endl;
    cout << "請選擇: ";
}

// 自訂參數生成
void custom_generation() {
    NoiseConfig config;
    
    cout << "\n=== 自訂參數設置 ===" << endl;
    
    config.width = get_input<int>("圖像寬度", 512);
    config.height = get_input<int>("圖像高度", 512);
    config.scale = get_input<double>("噪聲縮放因子", 5.0);
    
    cout << "\n觀察平面設置:" << endl;
    cout << "x - YZ平面 (固定X座標)" << endl;
    cout << "y - XZ平面 (固定Y座標)" << endl; 
    cout << "z - XY平面 (固定Z座標)" << endl;
    config.plane = get_input<char>("選擇觀察平面 (x/y/z)", 'z');
    config.plane_position = get_input<double>("平面位置", 0.0);
    
    config.use_turbulence = get_yes_no("使用湍流噪聲", false);
    config.normalize = get_yes_no("標準化噪聲值", true);
    
    config.output_name = get_input<string>("輸出檔案名前綴", "perlin_custom");
    
    generate_perlin_image(config);
}

// 快速預設生成
void quick_presets() {
    cout << "\n=== 快速預設選擇 ===" << endl;
    cout << "1. 基礎Perlin噪聲" << endl;
    cout << "2. 湍流噪聲" << endl;
    cout << "3. 大尺度噪聲" << endl;
    cout << "4. 小尺度噪聲" << endl;
    cout << "5. 高解析度" << endl;
    
    int choice = get_input<int>("選擇預設", 1);
    
    NoiseConfig config;
    
    switch (choice) {
        case 1:
            config.output_name = "preset_basic";
            config.scale = 5.0;
            break;
        case 2:
            config.output_name = "preset_turbulence";
            config.use_turbulence = true;
            config.scale = 3.0;
            break;
        case 3:
            config.output_name = "preset_large_scale";
            config.scale = 20.0;
            break;
        case 4:
            config.output_name = "preset_small_scale";
            config.scale = 1.0;
            break;
        case 5:
            config.output_name = "preset_hires";
            config.width = 1024;
            config.height = 1024;
            config.scale = 8.0;
            break;
        default:
            cout << "無效選擇，使用基礎預設" << endl;
            config.output_name = "preset_basic";
            break;
    }
    
    generate_perlin_image(config);
}

// 生成動畫序列
void generate_animation() {
    cout << "\n=== 動畫序列生成 ===" << endl;
    
    NoiseConfig config;
    config.width = get_input<int>("圖像寬度", 256);
    config.height = get_input<int>("圖像高度", 256);
    config.scale = get_input<double>("噪聲縮放因子", 5.0);
    config.use_turbulence = get_yes_no("使用湍流噪聲", false);
    
    int num_frames = get_input<int>("畫格數", 10);
    double start_pos = get_input<double>("起始平面位置", 0.0);
    double end_pos = get_input<double>("結束平面位置", 10.0);
    
    cout << "\n正在生成 " << num_frames << " 格動畫序列..." << endl;
    
    for (int i = 0; i < num_frames; i++) {
        config.plane_position = start_pos + (end_pos - start_pos) * i / (num_frames - 1);
        config.output_name = "animation_frame_" + to_string(i);
        
        cout << "\n=== 生成第 " << (i + 1) << "/" << num_frames 
             << " 格 (位置=" << config.plane_position << ") ===" << endl;
        
        generate_perlin_image(config);
    }
    
    cout << "\n動畫序列生成完成！" << endl;
    cout << "可以使用以下指令創建GIF動畫:" << endl;
    cout << "convert -delay 20 animation_frame_*.png animation.gif" << endl;
}

int main() {
    while (true) {
        show_menu();
        
        int choice = get_input<int>("", 1);
        
        switch (choice) {
            case 1:
                custom_generation();
                break;
            case 2:
                quick_presets();
                break;
            case 3:
                generate_animation();
                break;
            case 4:
                cout << "再見！" << endl;
                return 0;
            default:
                cout << "無效選擇，請重試。" << endl;
                break;
        }
    }
    
    return 0;
} 