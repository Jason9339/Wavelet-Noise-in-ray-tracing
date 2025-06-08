#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "noise_utils.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "\n=== 生成各種噪聲圖像 ===" << std::endl;

    // 1. 純 2D 噪聲
    std::cout << "1. 生成純 2D 噪聲..." << std::endl;
    generate_noise_2d_image(512, 512, 1.0, 20.0, "noise_2d_01.png");

    // 2. 水平平面 y=0
    std::cout << "\n2. 生成 3D 噪聲水平平面切片 (y=0)..." << std::endl;
    generate_noise_3d_plane_image(512, 512, 1.0, 20.0,
        point3(0, 0, 0), vec3(0, 1, 0), "noise_3d_y_02.png");

    // 3. 垂直平面 x=0
    std::cout << "\n3. 生成 3D 噪聲垂直平面切片 (x=0)..." << std::endl;
    generate_noise_3d_plane_image(512, 512, 1.0, 20.0,
        point3(0, 0, 0), vec3(1, 0, 0), "noise_3d_x_03.png");

    // 4. 斜面
    std::cout << "\n4. 生成 3D 噪聲斜面切片..." << std::endl;
    generate_noise_3d_plane_image(512, 512, 1.0, 20.0,
        point3(0, 0, 0), vec3(1, 0.2, 0.7), "noise_3d_diag_04.png");

    // 5. 高頻 2D
    std::cout << "\n5. 生成高頻 2D 噪聲..." << std::endl;
    generate_noise_2d_image(256, 256, 4.0, 20.0, "noise_2d_hf_05.png");

    std::cout << "\n=== 所有測試完成！===" << std::endl;
    std::cout << "請檢查 ./noise_*.png 檔案" << std::endl;
    return 0;
}
