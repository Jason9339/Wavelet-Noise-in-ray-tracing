# Wavelet Noise in Ray Tracing

這個項目實現了經典的 Perlin 噪聲演算法，並將其應用於光線追踪渲染器中。項目包含 Perlin 噪聲實現和未完成的小波噪聲探索，用於光線追踪應用。

## 功能特點

- **完整的 Perlin 噪聲實現** - 支援 2D 和 3D 噪聲生成
- **分形噪聲（Fractal Noise）** - 多層疊加產生複雜的噪聲模式
- **3D 平面切片** - 從 3D 噪聲場中提取任意平面的切片
- **光線追踪應用** - 將噪聲用作材質紋理進行真實感渲染
- **PNG 圖像輸出** - 高品質的圖像格式輸出
- **頻率分析** - Python notebook 分析噪聲頻譜特性

## 系統需求

- **C++ 編譯器**：支援 C++17 標準（如 `g++` 或 `clang++`）
- **Make** 工具用於編譯
- **Python 3**（可選）用於執行分析 notebook
- Python 套件可透過 `requirements.txt` 安裝：
  ```bash
  pip install -r requirements.txt
  ```

C++ 程式無需額外函式庫，僅依賴提供的原始碼檔案和標頭檔 `stb_image_write`。

## 項目結構

### 主要程式
- **`test_noise`** - 噪聲測試程式，生成各種噪聲圖像樣本
- **`main`** - 光線追踪渲染器，使用噪聲紋理渲染 3D 場景

### 核心檔案
- `perlin.h` - Perlin 噪聲核心實現
- `noise_utils.h/cpp` - 噪聲圖像生成工具函數
- `texture.h` - 紋理系統，包含噪聲紋理
- `material.h` - 材質系統
- `main.cpp` - 光線追踪主程式
- `test_noise.cpp` - 噪聲測試主程式

### 分析工具
- `analysis/perlin_frequency_analysis.ipynb` - Perlin 噪聲頻率分析 notebook

## 編譯與執行

### 編譯所有程式
```bash
make all
```
這會產生兩個可執行檔：
- `test_noise` – 生成 Perlin 噪聲圖像
- `main` – 光線追踪程式

### 單獨編譯
```bash
make test_noise    # 編譯噪聲測試程式
make main          # 編譯光線追踪程式
```

### 執行程式

#### 生成噪聲測試圖像
```bash
make run_test
# 或直接執行
./test_noise
```

#### 執行光線追踪渲染
```bash
make run_main  
# 或直接執行
./main
```

使用 `make clean` 清理執行檔和生成的圖像。

## 噪聲測試程式功能

`test_noise` 程式會自動生成以下噪聲圖像：

### 1. 純 2D 噪聲 (`noise_2d_01.png`)
- **尺寸**: 512×512
- **說明**: 使用 2D 分形噪聲生成的基礎噪聲圖案
- **縮放**: 1.0

### 2. 3D 噪聲水平切片 (`noise_3d_y_02.png`)
- **切片平面**: y = 0 (水平平面)
- **說明**: 從 3D 噪聲場中切出的水平截面

### 3. 3D 噪聲垂直切片 (`noise_3d_x_03.png`)
- **切片平面**: x = 0 (垂直平面)
- **說明**: 從 3D 噪聲場中切出的垂直截面

### 4. 3D 噪聲斜面切片 (`noise_3d_diag_04.png`)
- **切片平面**: 法向量 (1, 0.2, 0.7)
- **說明**: 展示任意角度平面切片的能力

### 5. 高頻 2D 噪聲 (`noise_2d_hf_05.png`)
- **尺寸**: 256×256
- **縮放**: 4.0 (高頻率，更細緻的細節)
- **說明**: 展示不同頻率的噪聲效果

## 光線追踪應用

`main` 程式創建一個 3D 場景，包含：

- **噪聲紋理地面** - 使用 Perlin 噪聲作為地面材質
- **噪聲紋理球體** - 展示噪聲在曲面上的效果
- **光源** - 提供場景照明
- **材質系統** - 支援漫反射、金屬、電介質等材質

### 輸出檔案
- `raytrace.png` - PNG 格式的渲染結果
- `raytrace.ppm` - PPM 格式的渲染結果

## 頻率分析

使用 Python notebook 分析 Perlin 噪聲的頻譜特性：

```bash
cd analysis
jupyter notebook perlin_frequency_analysis.ipynb
```

此分析展示 Perlin 噪聲的 2D 傅里葉頻譜，驗證它不是帶限的，這對於理解噪聲在光線追踪中的別名效應很重要。

## 清理與維護

```bash
make clean         # 清理執行檔和所有圖像
make clean_images  # 只清理生成的圖像檔案
make help          # 顯示所有可用命令
```

## 相依性

- **STB Image Write** - 用於 PNG 圖像輸出（已包含）
- **C++ 標準庫** - 數學函數和 I/O 操作
- **matplotlib, pillow, numpy** - Python 分析工具（可選）
- **無外部相依性** - C++ 項目完全自包含