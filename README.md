[[Traditional Chinese](#繁體中文)] [[English](README-en.md)]

# Wavelet Noise in Ray Tracing

本專案整合了小波噪聲與光線追蹤技術，提供完整的研究與可視化工具。

---

## 第一部分：光線追蹤渲染

### 功能說明
用 ray tracing 方式呈現 wavelet 3D noise texture 和 Perlin 3D noise texture

### 使用方式
```bash
# 編譯專案
make

# 單一結果
make run

# 一次生成兩個結果
make compare
```

### 渲染結果展示

| Perlin Noise 光線追蹤渲染結果 | Wavelet 3D Noise 光線追蹤渲染結果 |
|:---:|:---:|
| <img src="result_raytracing/raytrace_Perlin_octave4.png" width="400" alt="Perlin Ray Tracing Result"> | <img src="result_raytracing/raytrace_Wavelet3D_octave4.png" width="400" alt="Wavelet 3D Ray Tracing Result"> |


---

## 第二部分：噪聲實驗與分析

### 功能說明
實驗產出接近原始論文 Figure 8 的結果，驗證 wavelet noise band-limite 的特性

### 原始論文參照
<img src="asset/wavelet%20noise%20figure8.png" width="500" alt="Wavelet Function Figure 8">

<table>
<tr>
<td width="200px"><strong>我的實驗結果 (Octave 4)</strong></td>
<td align="center"></td>
</tr>
<tr>
<td width="200px"><strong>Perlin Noise 分析:</strong></td>
<td></td>
</tr>
<tr>
<td width="200px">Perlin 2D Noise 詳細分析 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/perlin_noise_2D_octave_4_detailed_analysis.png" width="450" alt="Perlin 2D Analysis"></td>
</tr>
<tr>
<td width="200px">Perlin 3D Sliced Noise 詳細分析 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/perlin_noise_3Dsliced_octave_4_detailed_analysis.png" width="450" alt="Perlin 3D Sliced Analysis"></td>
</tr>
<tr>
<td width="200px"><strong>Wavelet Noise 分析(此為單一頻帶結果):</strong></td>
<td></td>
</tr>
<tr>
<td width="200px">Wavelet 2D Noise 詳細分析 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/wavelet_noise_2D_octave_4_detailed_analysis.png" width="450" alt="Wavelet 2D Analysis"></td>
</tr>
<tr>
<td width="200px">Wavelet 3D Sliced Noise 詳細分析 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/wavelet_noise_3Dsliced_octave_4_detailed_analysis.png" width="450" alt="Wavelet 3D Sliced Analysis"></td>
</tr>
<tr>
<td width="200px">Wavelet 3D Projected Noise 詳細分析 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/wavelet_noise_3Dprojected_octave_4_detailed_analysis.png" width="450" alt="Wavelet 3D Projected Analysis"></td>
</tr>
<tr>
<td width="200px"><strong>整體比較分析:</strong></td>
<td></td>
</tr>
<tr>
<td width="200px">Wavelet Noise 完整比較 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/wavelet_full_comparison_octave_4.png" width="450" alt="Wavelet Full Comparison"></td>
</tr>
<tr>
<td width="200px">與原始論文 Figure 8 的比較 (Octave 4)</td>
<td align="center"><img src="experient/result_analyze/figure8_comparison_octave_4.png" width="450" alt="Figure 8 Comparison"></td>
</tr>
</table>


### 使用方式
```bash
# 進入實驗目錄並編譯
cd experient && make

# 執行噪聲實驗（生成 raw 數據）
cd experient && make run

# 分析實驗結果
cd experient && python3 analyze.py
```

---

## 第三部分：Web 可視化

### 功能說明
提供互動式 Web 介面，讓使用者能夠即時觀察和操作噪聲效果。

### 使用方式
```bash
# 轉換 raw 數據為 JSON 格式
cd threejs && python3 convert_raw_to_json.py --batch

# 啟動 Web 伺服器
cd threejs && python3 -m http.server 8000

# 在瀏覽器中開啟 http://localhost:8000
```

### Web 可視化結果展示
*(原始版本具有互動功能)*

| Perlin Noise 互動式可視化介面 | Wavelet Noise 互動式可視化介面 |
|:---:|:---:|
| <img src="asset/threejs_perlin.png" width="400" alt="Perlin Noise Web Visualization"> | <img src="asset/threejs_wavelet.png" width="400" alt="Wavelet Noise Web Visualization"> |