# Perlin 噪聲灰度圖生成器

這個項目提供了兩個程式來生成 Perlin 噪聲的灰度圖像，支援指定觀察平面。

## 功能特點

- 🎨 生成高品質的 Perlin 噪聲灰度圖
- 📐 支援選擇不同的觀察平面 (XY, XZ, YZ)
- 🌪️ 支援湍流噪聲效果
- 📏 可調節噪聲縮放因子
- 💾 同時輸出 PNG 和 PPM 格式
- 🎬 支援生成動畫序列

## 程式說明

### 1. `perlin_viewer` - 自動化生成器
自動生成多種預設的 Perlin 噪聲圖像，包括：
- 基礎 Perlin 噪聲
- 湍流噪聲
- 不同尺度的噪聲
- Z軸切片序列
- 高解析度圖像

### 2. `perlin_interactive` - 互動式生成器
提供選單式介面，允許使用者：
- 自訂所有參數
- 選擇快速預設
- 生成動畫序列

## 編譯和執行

### 編譯所有程式
```bash
make all
```

### 單獨編譯
```bash
make perlin_viewer      # 編譯自動化生成器
make perlin_interactive # 編譯互動式生成器
```

### 執行程式
```bash
make run_viewer         # 執行自動化生成器
make run_interactive    # 執行互動式生成器
```

或者直接執行：
```bash
./perlin_viewer         # 自動生成多種預設圖像
./perlin_interactive    # 互動式介面
```

## 參數說明

### 基礎參數
- **圖像尺寸**: 設定輸出圖像的寬度和高度
- **噪聲縮放因子**: 控制噪聲的細節程度，值越大細節越少
- **觀察平面**: 
  - `x` - YZ平面 (固定X座標)
  - `y` - XZ平面 (固定Y座標)  
  - `z` - XY平面 (固定Z座標)
- **平面位置**: 在指定軸上的切片位置

### 噪聲類型
- **基礎噪聲**: 標準的 Perlin 噪聲
- **湍流噪聲**: 多層噪聲疊加，產生更複雜的圖案

### 特殊效果
- **標準化**: 將噪聲值標準化到 [0,1] 範圍

## 輸出檔案

程式會生成以下檔案：
- `*.png` - PNG格式圖像（推薦）
- `*.ppm` - PPM格式圖像（便於處理）

## 使用範例

### 互動式生成自訂圖像
```bash
./perlin_interactive
# 選擇 "1. 自訂參數生成"
# 按提示輸入參數
```

### 生成動畫序列
```bash
./perlin_interactive
# 選擇 "3. 生成動畫序列"
# 設定畫格數和位置範圍
```

生成動畫後，可以使用 ImageMagick 創建 GIF：
```bash
convert -delay 20 animation_frame_*.png animation.gif
```

## 預設效果

自動化生成器會產生以下預設：

1. **基礎 Perlin 噪聲** (`perlin_basic.png`)
   - 標準的平滑噪聲圖案

2. **湍流噪聲** (`perlin_turbulence.png`)
   - 複雜的多層噪聲效果

3. **不同尺度** (`perlin_scale_*.png`)
   - 展示不同縮放因子的效果

4. **Z軸切片** (`perlin_slice_z*.png`)
   - 不同Z位置的切片圖像

5. **高解析度** (`perlin_hires_turbulence.png`)
   - 1024x1024 高解析度湍流噪聲

## 清理檔案

```bash
make clean        # 清理執行檔和圖像
make clean_images # 只清理圖像檔案
```

## 技術實作

- 使用 C++17 標準
- 基於 Ken Perlin 的經典演算法實作
- 支援三線性插值和平滑過渡
- 使用 stb_image_write 函式庫輸出 PNG 格式

## 相依檔案

- `perlin.h` - Perlin 噪聲核心實作
- `vec3.h` - 3D向量數學函式庫
- `rtweekend.h` - 基礎工具函數
- `stb_image_write.h` - 圖像輸出函式庫

## 注意事項

- 高解析度圖像生成可能需要較長時間
- 動畫序列會生成多個檔案，注意磁碟空間
- 推薦使用 PNG 格式，品質更好且檔案更小 