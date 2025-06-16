# Wavelet Noise in Ray Tracing

這個項目實現了 Wavelet Noise 在光線追踪中的應用，展示了帶限噪聲相對於傳統 Perlin noise 的優勢。

## 快速開始

### 編譯
```bash
make
```

### 執行
```bash
make run
```

### 生成比較
```bash
make compare
```

### 清理
```bash
make clean
```

## 使用說明

### 交互式執行
運行程序後會提示選擇：
- 噪聲類型：0 (Perlin) 或 1 (Wavelet 3D)
- Octave 級別：3-5 (預設為 4)

### 輸出檔案
渲染結果保存在 `result_raytracing/` 目錄：
- `raytrace_Perlin_octave4.png`
- `raytrace_Wavelet3D_octave4.png`

### 實驗數據
- 生成實驗數據：`cd experient && ./wavelet_noise`
- 分析結果：`cd experient && python3 analyze.py`
- Web 可視化：`cd threejs && python3 -m http.server 8000`

## 項目結構

```
├── main.cpp                    # 光線追踪主程式
├── Makefile                    # 編譯配置
├── WaveletNoise.h/cpp          # Wavelet noise 實現
├── perlin.h                    # Perlin noise 實現
├── texture.h                   # 噪聲紋理類
├── experient/                  # 實驗和分析模組
├── threejs/                    # Web 可視化模組
└── result_raytracing/          # 光線追踪渲染結果
```

## 系統需求

- C++ 編譯器 (支援 C++17)
- Make
- Python 3 (用於實驗分析)

---

# Wavelet Noise in Ray Tracing

This project implements Wavelet Noise in ray tracing, demonstrating the advantages of band-limited noise over traditional Perlin noise.

## Quick Start

### Compile
```bash
make
```

### Run
```bash
make run
```

### Generate Comparison
```bash
make compare
```

### Clean
```bash
make clean
```

## Usage

### Interactive Execution
The program will prompt you to choose:
- Noise type: 0 (Perlin) or 1 (Wavelet 3D)
- Octave level: 3-5 (default: 4)

### Output Files
Rendered results are saved in `result_raytracing/` directory:
- `raytrace_Perlin_octave4.png`
- `raytrace_Wavelet3D_octave4.png`

### Experimental Data
- Generate experimental data: `cd experient && ./wavelet_noise`
- Analyze results: `cd experient && python3 analyze.py`
- Web visualization: `cd threejs && python3 -m http.server 8000`

## Project Structure

```
├── main.cpp                    # Ray tracing main program
├── Makefile                    # Build configuration
├── WaveletNoise.h/cpp          # Wavelet noise implementation
├── perlin.h                    # Perlin noise implementation
├── texture.h                   # Noise texture classes
├── experient/                  # Experiment and analysis module
├── threejs/                    # Web visualization module
└── result_raytracing/          # Ray tracing render results
```

## Requirements

- C++ compiler (C++17 support)
- Make
- Python 3 (for experimental analysis)
