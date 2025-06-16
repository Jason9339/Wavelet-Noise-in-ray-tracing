# Wavelet Noise in Ray Tracing

這個項目實現了經典的 Perlin 噪聲演算法，並將其應用於光線追踪渲染器中。該項目專注於光線追踪應用，使用噪聲作為材質紋理進行真實感渲染。

## 功能特點

- **完整的 Perlin 噪聲實現** - 支援 2D 和 3D 噪聲生成
- **分形噪聲（Fractal Noise）** - 多層疊加產生複雜的噪聲模式
- **光線追踪應用** - 將噪聲用作材質紋理進行真實感渲染
- **PNG 圖像輸出** - 高品質的圖像格式輸出

## 系統需求

- **C++ 編譯器**：支援 C++17 標準（如 `g++` 或 `clang++`）
- **Make** 工具用於編譯

C++ 程式無需額外函式庫，僅依賴提供的原始碼檔案和標頭檔 `stb_image_write`。

## 項目結構

### 主要程式
- **`main`** - 光線追踪渲染器，使用噪聲紋理渲染 3D 場景

### 核心檔案
- `perlin.h` - Perlin 噪聲核心實現
- `texture.h` - 紋理系統，包含噪聲紋理
- `material.h` - 材質系統
- `main.cpp` - 光線追踪主程式

### 其他支援檔案
- `vec3.h`, `ray.h`, `sphere.h`, `quad.h` - 光線追踪核心類別
- `hittable.h`, `hittable_list.h` - 物體碰撞檢測
- `color.h`, `interval.h` - 顏色和區間工具
- `stb_image_write.h` - PNG 圖像輸出

## 編譯與執行

### 編譯程式
```bash
make all
# 或
make main
```

### 執行光線追踪渲染
```bash
make run  
# 或直接執行
./main
```

使用 `make clean` 清理執行檔和生成的圖像。

## 光線追踪應用

`main` 程式創建一個 3D 場景，包含：

- **噪聲紋理地面** - 使用 Perlin 噪聲作為地面材質
- **噪聲紋理球體** - 展示噪聲在曲面上的效果
- **光源** - 提供場景照明
- **材質系統** - 支援漫反射、金屬、電介質等材質

### 輸出檔案
- `raytrace.png` - PNG 格式的渲染結果
- `raytrace.ppm` - PPM 格式的渲染結果

## 清理與維護

```bash
make clean         # 清理執行檔和所有圖像
make clean_images  # 只清理生成的圖像檔案
make help          # 顯示所有可用命令
```

## 相依性

- **STB Image Write** - 用於 PNG 圖像輸出（已包含）
- **C++ 標準庫** - 數學函數和 I/O 操作
- **無外部相依性** - C++ 項目完全自包含