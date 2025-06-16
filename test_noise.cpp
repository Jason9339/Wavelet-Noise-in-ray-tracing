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
    int tile_s = 128; // Default tile size for wavelet noise generators

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

    // --- In test_noise.cpp, inside main() before return 0; ---

    std::cout << "\n\n=== 生成單一 Tile Wavelet 噪聲圖像 (128x128 Tile, 無重複) ===" << std::endl;

    int generator_tile_size = 128;
    int image_to_render_size = 128; // Match generator_tile_size for 1:1 view of the tile

    std::cout << "\n Test Case: 2D 單一 Tile (1:1 sampling, GenTile: 128x128)..." << std::endl;
    generate_wavelet_noise_2d_single_tile_image(
        image_to_render_size, 
        1.0,  // <<<--- CRITICAL: Sampling density 1.0 for non-repeating view of one tile
        "result/wavelet_2d_true_single_tile_sds1_ts128.png", 
        generator_tile_size
    );

    // If you want to see higher "frequency" content *from within* that single conceptual tile,
    // you would need to modify WNoise2D itself or how it uses its tile_data_2d.
    // The current `sampling_density_scale` > 1 will always cause tiling due to Mod.

    // Let's try a different interpretation: what if sampling_density_scale means
    // how "zoomed in" we are on the tile data?
    // To see a "higher frequency" version of a *single non-repeating tile*,
    // you'd need the base noise band itself to be of higher frequency.
    // The `WNoise2D` function, as defined by the paper, produces *one specific band* of noise.
    // `sampling_density_scale` effectively scales the input coordinates to this one band.

    // The image you showed (ics4_ts128) is actually a valid FFT for an image that IS spatially periodic.
    // To get the FFT of *one tile* of the band-limited noise:
    // 1. image_size = tile_size_for_generator (e.g., 128)
    // 2. sampling_density_scale = 1.0

    // If you want to see the spectrum for what corresponds to a "higher frequency band"
    // in a multi-band fractal sum, but still just one band's FFT:
    // You would typically use the same `WNoise2D` but scale its input coordinates *before* passing them.
    // This is what `generate_wavelet_noise_2d_single_band_image` does with its `scale` parameter.
    // The key difference is that `generate_wavelet_noise_2d_single_band_image` samples over a `world_size`
    // which might be larger or smaller than one tile, and then the `scale` parameter is applied.

    // Let's re-run the original single_band_image with explicit 128x128 image size and carefully chosen scale:
    std::cout << "\n Re-test: Original Single Band (Image: 128x128, World: 128, Scale: 1.0, GenTile: 128)..." << std::endl;
    generate_wavelet_noise_2d_single_band_image(
        128, 128, // width, height
        1.0,     // scale factor for coordinates
        128.0,   // world_size (make it match tile size)
        "result/wavelet_2d_single_band_s1_w128_ts128.png",
        128      // tile_size for generator
    );
    // In this case, p_sample.x = (((i/128)-0.5)*128 + offset.x)*1.0 = (i-64 + offset.x)
    // This still allows sampling across tile boundaries due to offset and -0.5.

    // The `generate_wavelet_noise_2d_single_tile_image` with `sampling_density_scale = 1.0`
    // and `image_size = tile_size_for_generator` is the most direct way to get an image
    // of one, non-repeated, uninterpolated (if you could access tile_data directly) or
    // minimally-interpolated-within-itself tile.
    // The previous `p_input_for_wnoise` logic for single_tile was this:
    // p_input_for_wnoise.e[0] = (static_cast<double>(i) / image_size) * image_size * sampling_density_scale;
    // This is IDENTICAL to:
    // p_input_for_wnoise.e[0] = static_cast<double>(i) * sampling_density_scale;
    // So if image_size=128, i goes 0..127. If sampling_density_scale=4, p_input.x goes 0..508.
    // This WILL cause tiling within WNoise2D if its internal tile_size is 128.

    // The image `wavelet_2d_single_tile_ics4_ts128.png` is an image of a 128x128 tile,
    // where the content is formed by taking the WNoise2D generator (which itself has a 128x128 period due to Mod)
    // and sampling it with coordinates that go effectively 4x across its period in x and 4x in y.
    // So the resulting 128x128 image *does* contain 4x4 repetitions of the fundamental 32x32 pattern
    // that arises when you sample a 128-periodic function with a 4x denser sampling rate.
    // Its FFT *should* be a grid, and it is.

    // What you likely want for "FFT of one band without tiling artifacts":
    // Call `generate_wavelet_noise_2d_single_tile_image` with `sampling_density_scale = 1.0`.
    // This will make `p_input_for_wnoise` range from `0` to `image_size-1`.
    // If `image_size` is also `tile_size_for_generator`, then `WNoise2D`'s `Mod` operation
    // will effectively be a no-op for these coordinates, and you'll get one non-repeating tile.
    // The "frequency" of this tile is the base frequency of the `N(x)` band.
    std::cout << "\n Test Case: True Single Non-Repeating Tile (128x128, SDS=1.0)..." << std::endl;
    generate_wavelet_noise_2d_single_tile_image(
        128, 
        1.0,
        "result/wavelet_2d_NON_REPEATING_tile_sds1_ts128.png", 
        128
    );
    // Now, to see a "higher frequency band" equivalent, but still non-repeating for FFT:
    // We can't just increase SDS if we want non-repeating.
    // We'd need to effectively change the *content* of the tile to be higher frequency,
    // OR take a smaller piece of the low-frequency tile and zoom in (but FFT size changes).

    // Let's re-evaluate the original single_band function's parameters:
    // `generate_wavelet_noise_2d_single_band_image(width, height, scale, world_size, filename, tile_size)`
    // `p_sample = (p_world + offset) * scale`
    // `p_world.x = ((i/width)-0.5)*world_size`
    // If width=128, world_size=128: p_world.x goes from -64 to 63.
    // If scale=1: p_sample.x goes from (-64+offset.x) to (63+offset.x).
    // This samples a 128-unit wide continuous segment. WNoise2D will tile this.
    // To make this non-repeating, `world_size * scale` should be `< tile_size_for_generator`.
    // And `width` should be chosen appropriately.

    std::cout << "\n Test Case: Single Band, scale to fit one tile period (Image:128x128, GenTile:128, World:128, Scale:0.99)..." << std::endl;
    generate_wavelet_noise_2d_single_band_image(
        128,    // width
        128,    // height
        0.999 / (128.0), // Effective scale to make world_size * scale map to tile_size_for_generator
                 // (i/W-0.5)*WS ranges over WS. So (i/W-0.5)*WS * S needs to range over TSG
                 // Let ((i/W)-0.5)*WS be X. X ranges [-WS/2, WS/2).
                 // We want X*S to be in [0, TSG).
                 // Let's set world_size = tile_size_for_generator = 128.
                 // Then p_world.x is [-64, 63).
                 // If we add 64, it's [0, 127). Then scale by `s`.
                 // This is getting complicated.
                 // The `generate_wavelet_noise_2d_single_tile_image` with SDS=1.0 is the cleanest.
        128.0, // Make this effectively tile_size_for_generator
        "result/wavelet_2d_single_band_ versucht_non_repeating_ts128.png",
        128      // tile_size_for_generator
    );
    // For the above, modify generate_wavelet_noise_2d_single_band_image:
    // vec2 p_world( (double(i)/width) * world_size, (double(j)/height) * world_size ); // range [0, world_size)
    // vec2 p_sample = p_world * scale; // remove offset for this test
    // If world_size = 128, scale = 0.999, then p_sample is [0, 127.xx). This should work.


    std::cout << "\n Test Case: True Single Non-Repeating Tile (64x64, SDS=1.0)..." << std::endl;
    generate_wavelet_noise_2d_single_tile_image(
        64, 
        1.0,
        "result/wavelet_2d_NON_REPEATING_tile_sds1_ts64.png", 
        64
    );


    std::cout << "\n=== 所有測試完成！===" << std::endl;
    return 0;
}