#define STB_IMAGE_WRITE_IMPLEMENTATION // Keep this at the top of one .cpp file
#include "stb_image_write.h"
#include "noise_utils.h"
#include <iostream>
#include <string>

// Make sure vec2.h is included if noise_utils.h doesn't pull it in for main
// #include "vec2.h" // Likely noise_utils.h already includes it.

int main() {
    std::cout << "\n=== 生成 Perlin 噪聲圖像 ===" << std::endl;

    // 1. 純 2D Perlin 噪聲
    std::cout << "1. 生成純 2D Perlin 噪聲..." << std::endl;
    generate_noise_2d_image(512, 512, 1.0, 20.0, "result/perlin_2d_01.png");

    // 2. 水平平面 y=0 (Perlin 3D slice)
    std::cout << "\n2. 生成 3D Perlin 噪聲水平平面切片 (y=0)..." << std::endl;
    generate_noise_3d_plane_image(512, 512, 1.0, 20.0,
        point3(0, 0, 0), vec3(0, 1, 0), "result/perlin_3d_y_slice_02.png");

    // 3. 垂直平面 x=0 (Perlin 3D slice)
    std::cout << "\n3. 生成 3D Perlin 噪聲垂直平面切片 (x=0)..." << std::endl;
    generate_noise_3d_plane_image(512, 512, 1.0, 20.0,
        point3(0, 0, 0), vec3(1, 0, 0), "result/perlin_3d_x_slice_03.png");

    // 4. 斜面 (Perlin 3D slice)
    std::cout << "\n4. 生成 3D Perlin 噪聲斜面切片..." << std::endl;
    generate_noise_3d_plane_image(512, 512, 1.0, 20.0,
        point3(0, 0, 0), vec3(1, 0.2, 0.7), "result/perlin_3d_diag_slice_04.png");

    // 5. 高頻 2D Perlin
    std::cout << "\n5. 生成高頻 2D Perlin 噪聲..." << std::endl;
    generate_noise_2d_image(256, 256, 4.0, 20.0, "result/perlin_2d_hf_05.png");

    std::cout << "\n\n=== 生成 Wavelet 噪聲圖像 ===" << std::endl;
    int tile_s = 32; // Default tile size for wavelet noise generators

    // 6. 純 2D Wavelet 噪聲 (Figure 8d style)
    std::cout << "\n6. 生成純 2D Wavelet 噪聲..." << std::endl;
    generate_wavelet_noise_2d_image(512, 512, 1.0, 20.0, "result/wavelet_2d_pure_06.png", 7, 0.5, tile_s);
    generate_wavelet_noise_2d_image(256, 256, 4.0, 10.0, "result/wavelet_2d_pure_hf_07.png", 7, 0.5, tile_s); // Higher freq

    // 7. 3D Wavelet 噪聲的 XY 切片 (Figure 8e style)
    std::cout << "\n7. 生成 3D Wavelet 噪聲 XY 切片 (Z=0)..." << std::endl;
    generate_wavelet_noise_3d_slice_image(512, 512, 1.0, 20.0, 
                                          point3(10.0, 20.0, 30.0), // noise_offset
                                          2, 0.0, // slice_axis = Z, slice_coord = 0
                                          "result/wavelet_3d_xy_slice_08.png", 7, 0.5, tile_s);
    generate_wavelet_noise_3d_slice_image(256, 256, 4.0, 10.0, 
                                          point3(10.0, 20.0, 30.0), 
                                          2, 0.0, 
                                          "result/wavelet_3d_xy_slice_hf_09.png", 7, 0.5, tile_s);


    // 8. 3D Wavelet 噪聲投影到 XY 平面 (Figure 8f style)
    std::cout << "\n8. 生成 3D Wavelet 噪聲投影到 XY 平面..." << std::endl;
    generate_wavelet_noise_3d_projected_image(512, 512, 1.0, 20.0,
                                           point3(0,0,0), vec3(0,0,1), // plane at origin, normal along Z
                                           point3(5.5, 15.5, 25.5),    // noise_offset
                                           "result/wavelet_3d_projected_xy_10.png", 7, 0.5, tile_s);
    generate_wavelet_noise_3d_projected_image(256, 256, 4.0, 10.0,
                                           point3(0,0,0), vec3(0,0,1),
                                           point3(5.5, 15.5, 25.5),
                                           "result/wavelet_3d_projected_xy_hf_11.png", 7, 0.5, tile_s);


    // 9. Figure 1 style comparison - Wavelet noise on a receding plane (projected)
    std::cout << "\n9. 生成 Wavelet 噪聲投影到遠平面 (Figure 1 style)..." << std::endl;
    // Simulating a plane receding in Z, viewed from origin looking down -Z
    // We'll sample points on the ZY plane (x=fixed_dist_from_camera) and project 3D noise onto it.
    // Or more simply, a plane at z=0 viewed from above, but with increasing scale factor for distance.
    // For simplicity, let's use a fixed plane and let 'scale' represent detail level.
    generate_wavelet_noise_3d_projected_image(512, 512, 0.5, 40.0, // Lower scale = more detail, larger world_size
                                           point3(0,0,0), vec3(0,0,1),
                                           point3(0,0,0), // No specific noise offset for this generic view
                                           "result/wavelet_3d_projected_fig1_style_distant_12.png", 7, 0.5, tile_s);
    generate_wavelet_noise_3d_projected_image(512, 512, 2.0, 40.0, // Higher scale = less detail
                                           point3(0,0,0), vec3(0,0,1),
                                           point3(0,0,0),
                                           "result/wavelet_3d_projected_fig1_style_closer_13.png", 7, 0.5, tile_s);


    std::cout << "\n\n=== 生成單一頻段 Wavelet 噪聲圖像 (用於頻譜分析) ===" << std::endl;

    // 10. 單一頻段 2D Wavelet 噪聲 (用於頻譜分析，應該看到中心黑暗的頻譜圖)
    std::cout << "\n10. 生成單一頻段 2D Wavelet 噪聲..." << std::endl;
    generate_wavelet_noise_2d_single_band_image(512, 512, 1.0, 20.0, "result/wavelet_2d_single_band_14.png", tile_s);
    generate_wavelet_noise_2d_single_band_image(256, 256, 4.0, 10.0, "result/wavelet_2d_single_band_hf_15.png", tile_s);

    // 11. 單一頻段 3D Wavelet 噪聲的 XY 切片
    std::cout << "\n11. 生成單一頻段 3D Wavelet 噪聲 XY 切片 (Z=0)..." << std::endl;
    generate_wavelet_noise_3d_single_band_slice_image(512, 512, 1.0, 20.0,
                                          point3(10.0, 20.0, 30.0), // noise_offset
                                          2, 0.0, // slice_axis = Z, slice_coord = 0
                                          "result/wavelet_3d_single_band_xy_slice_16.png", tile_s);

    // 12. 單一頻段 3D Wavelet 噪聲投影到 XY 平面
    std::cout << "\n12. 生成單一頻段 3D Wavelet 噪聲投影到 XY 平面..." << std::endl;
    generate_wavelet_noise_3d_single_band_projected_image(512, 512, 1.0, 20.0,
                                           point3(0,0,0), vec3(0,0,1), // plane at origin, normal along Z
                                           point3(5.5, 15.5, 25.5),    // noise_offset
                                           "result/wavelet_3d_single_band_projected_xy_17.png", tile_s);

    std::cout << "\n13. 生成用於對比的單一頻段 Wavelet 噪聲（不同尺度）..." << std::endl;
    // 生成不同尺度的單一頻段噪聲用於頻譜分析對比
    generate_wavelet_noise_2d_single_band_image(512, 512, 0.5, 20.0, "result/wavelet_2d_single_band_low_scale_18.png", tile_s);  // 低頻
    generate_wavelet_noise_2d_single_band_image(512, 512, 2.0, 20.0, "result/wavelet_2d_single_band_mid_scale_19.png", tile_s);  // 中頻
    generate_wavelet_noise_2d_single_band_image(512, 512, 8.0, 20.0, "result/wavelet_2d_single_band_high_scale_20.png", tile_s); // 高頻


    std::cout << "\n=== 所有測試完成！===" << std::endl;
    std::cout << "請檢查 ./result/ 目錄中的圖片檔案" << std::endl;
    std::cout << "\n提示：" << std::endl;
    std::cout << "- 多頻段圖像 (06-13) 適合用於一般紋理渲染" << std::endl;
    std::cout << "- 單一頻段圖像 (14-20) 適合用於頻譜分析，其頻譜圖中心應該是黑暗的" << std::endl;
    return 0;
}