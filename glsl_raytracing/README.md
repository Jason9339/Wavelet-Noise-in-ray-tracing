# GLSL Ray Tracing 演示

使用 WebGL 和 GLSL 著色器實現的即時 Ray Tracing 演示專案。

## 功能特色

- 🔮 **即時 Ray Tracing**: 使用 GLSL fragment shader 實現的 Ray Marching
- 🏀 **3D 幾何體**: 包含反射金屬球體和紋理地板
- 🌊 **Perlin 雜訊紋理**: 動態產生的地板紋理
- 💡 **物理光照**: 包含環境光、漫反射和鏡面反射
- 🎮 **互動控制**: 滑鼠、鍵盤和UI滑桿控制
- 📊 **效能監控**: 即時FPS和幀時間顯示

## 技術實現

### Ray Marching
- 使用 Signed Distance Functions (SDF) 描述幾何體
- 球體 SDF: `length(p - center) - radius`
- 平面 SDF: `dot(p, normal) + distance`

### Perlin 雜訊
- 基於梯度的程序化雜訊產生
- 多層次雜訊 (Fractal Brownian Motion)
- 用於產生自然外觀的地板紋理

### 光照模型
- **環境光**: 基礎照明
- **漫反射**: Lambert 光照模型
- **鏡面反射**: Phong 反射模型
- **反射**: 地板表面的簡單反射

## 使用說明

### 啟動專案
1. 在 `glsl_raytracing` 資料夾中啟動本地伺服器
2. 開啟瀏覽器存取 `index.html`

### 控制方式

#### 滑鼠控制
- **拖曳**: 移動相機位置
- **滾輪**: 縮放相機距離

#### 鍵盤控制
- `W/S`: 前後移動相機
- `A/D`: 左右移動相機
- `Q/E`: 上下移動相機
- `R`: 重置相機位置

#### UI 控制
- **相機 X/Y/Z**: 調整相機位置
- **球體 Y**: 調整球體高度
- **雜訊頻率**: 調整地板紋理細節
- **重置按鈕**: 恢復預設設定

## 檔案結構

```
glsl_raytracing/
├── index.html          # 主頁面
├── shader-utils.js     # WebGL 工具和著色器原始碼
├── perlin-noise.js     # Perlin 雜訊和工具類
├── main.js            # 主應用邏輯
└── README.md          # 專案說明
```

## 技術要求

- 支援 WebGL 的現代瀏覽器
- 推薦使用 Chrome、Firefox 或 Safari
- 需要較新的顯示卡以獲得最佳效能

## 著色器詳解

### Vertex Shader
簡單的全螢幕四邊形渲染，將螢幕座標轉換為UV座標。

### Fragment Shader
實現了完整的 Ray Tracing 管線：

1. **射線產生**: 從相機位置發射射線
2. **場景求交**: 使用 Ray Marching 查找最近交點
3. **材質計算**: 根據物體類型確定材質屬性
4. **光照計算**: 應用 Phong 光照模型
5. **反射處理**: 計算地板反射
6. **色彩輸出**: 應用色調映射和伽馬校正

## 效能最佳化

- **自適應步進**: 根據距離調整 Ray Marching 步長
- **早期終止**: 距離閾值檢測避免無意義計算
- **LOD 控制**: 遠距離物體使用較少的取樣
- **著色器最佳化**: 減少分支和複雜數學運算

## 擴展可能

- 新增更多幾何體 (立方體、圓環等)
- 實現軟陰影和全域光照
- 新增後處理效果 (景深、光暈等)
- 支援動畫和粒子系統
- 實現體積渲染 (雲、霧等)

## 問題排除

### 黑屏問題
- 檢查瀏覽器主控台是否有著色器編譯錯誤
- 確認 WebGL 支援是否啟用
- 嘗試降低渲染解析度

### 效能問題
- 降低 MAX_STEPS 常數值
- 減少反射計算複雜度
- 使用較小的畫布尺寸

### 相容性問題
- 某些行動裝置可能不支援高精度浮點運算
- 較老的顯示卡可能無法執行複雜的 fragment shader

## 參考資料

- [Ray Marching 教學](https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/)
- [Inigo Quilez SDF 函數庫](https://iquilezles.org/articles/distfunctions/)
- [Perlin 雜訊原理](https://en.wikipedia.org/wiki/Perlin_noise)
- [WebGL 基礎教學](https://webglfundamentals.org/)

---

這個專案演示了現代 GPU 著色器程式設計的強大能力，以及如何在 Web 平台上實現高品質的即時圖形渲染。 