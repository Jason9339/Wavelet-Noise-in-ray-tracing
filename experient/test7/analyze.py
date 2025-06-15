import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from mpl_toolkits.mplot3d import Axes3D
import os

def load_raw_data(filename, size=512):
    """Loads a raw float32 file into a numpy array."""
    try:
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        expected_size = size * size
        if data.size != expected_size:
            print(f"Warning: Size mismatch in {filename}. Expected {expected_size}, got {data.size}.")
            return None
        return data.reshape((size, size))
    except FileNotFoundError:
        print(f"Warning: File not found - {filename}")
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def plot_noise_column(ax_col, data, title):
    """Plots a full analysis column (pattern, fft, radial) for given data."""
    if data is None:
        # Handle case where data failed to load
        for ax in ax_col:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
            ax.axis('off')
        ax_col[0].set_title(title, fontsize=12)
        return

    # --- Calculations ---
    data_centered = data - np.mean(data)
    F = fftshift(fft2(data_centered))
    power_spectrum = np.abs(F) ** 2

    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    max_r = min(cx, cy)
    radial_profile = []
    for r in range(max_r):
        mask = (R >= r) & (R < r + 1)
        if np.any(mask):
            radial_profile.append(np.mean(power_spectrum[mask]))
        else:
             radial_profile.append(0)

    freqs = np.arange(len(radial_profile)) / (2 * len(radial_profile))
    
    # --- Plotting ---
    # 1. Noise Pattern
    ax = ax_col[0]
    ax.imshow(data, cmap='gray')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    # 2. Power Spectrum (log scale) - <<< 修改點在這裡
    ax = ax_col[1]
    power_log = np.log1p(power_spectrum.copy()) # 使用 copy 避免修改原始數據
    
    # 將中心點(DC分量)的能量設為0，以便更好地計算顏色範圍
    power_log[cy, cx] = 0
    
    # 使用百分位數來避免極端值，但現在的計算基於非DC分量
    vmax = np.percentile(power_log, 99.9) 
    # 稍微提高對比度，讓背景更黑，頻帶更亮
    vmin = vmax * 0.1 
    
    ax.imshow(power_log, cmap='hot', vmin=vmin, vmax=vmax)
    ax.axis('off')

    # 3. Radial Profile
    ax = ax_col[2]
    ax.semilogy(freqs, radial_profile, linewidth=1.5)
    ax.set_xlim(0, 0.5)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Frequency')
    peak_freq = freqs[np.argmax(radial_profile)]
    ax.axvline(peak_freq, color='r', linestyle='--', linewidth=1, label=f'Peak ≈ {peak_freq:.3f}')
    ax.legend(fontsize=8)
    
    # Set y-label only for the first column
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel('Power')

def create_comparison_figure(two_d_files, three_d_files):
    """Generates the main comparison figure for 2D vs 3D sliced noise.
    
    Args:
        two_d_files: 2D噪声文件列表
        three_d_files: 3D切片噪声文件列表
    """
    # 我們總共有 3 (2D) + 1 (間隔) + 3 (3D) = 7 欄
    num_cols = len(two_d_files) + 1 + len(three_d_files)
    num_rows = 3
    
    # 創建一個 3 行, 7 欄的網格
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 9),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.2, 1, 1, 1], 'wspace': 0.5})
    
    plt.suptitle('Wavelet Noise: Pure 2D vs. 3D Sliced Comparison', fontsize=18, y=0.98)

    # --- Pure 2D Noise (Top Row of Conceptual Blocks) ---
    for i, filename in enumerate(two_d_files):
        data = load_raw_data(filename)
        octave = filename.split('_')[-1].replace('.raw', '')
        title = f"Pure 2D - Octave {octave}"
        plot_noise_column(axes[:, i], data, title)

    # --- 3D Sliced Noise (Bottom Row of Conceptual Blocks) ---
    # 將中間的第 4 欄 (索引為 3) 設為不可見，作為間隔
    for row in range(num_rows):
        axes[row, len(two_d_files)].axis('off')

    for i, filename in enumerate(three_d_files):
        # 3D 噪聲的圖放在間隔欄之後
        col_idx = i + len(two_d_files) + 1
        data = load_raw_data(filename)
        octave = filename.split('_')[-1].replace('.raw', '')
        title = f"3D Sliced - Octave {octave}"
        plot_noise_column(axes[:, col_idx], data, title)
        
    fig.text(0.26, 0.93, 'Pure 2D Wavelet Noise', ha='center', va='center', fontsize=14, weight='bold')
    fig.text(0.74, 0.93, '3D Sliced Wavelet Noise', ha='center', va='center', fontsize=14, weight='bold')

    # 調整佈局以避免重疊
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    # 由於 tight_layout 可能會覆蓋 wspace，我們再手動調整一次
    plt.subplots_adjust(wspace=0.5)

    plt.savefig("wavelet_2D_vs_3D_comparison.png", dpi=150)
    plt.show()

def analyze_single_band(filename, title):
    """Analyze a single band noise pattern to show band-limited property"""
    try:
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        if data.size != 512 * 512:
            print(f"Size mismatch in {filename}: expected 262144 elements, got {data.size}")
            return None
        data = data.reshape((512, 512))
    except:
        print(f"Could not read {filename}")
        return None

    data_centered = data - np.mean(data)
    F = fftshift(fft2(data_centered))
    magnitude = np.abs(F)
    power_spectrum = magnitude ** 2

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(title, fontsize=14, y=1.02)

    ax = axes[0]
    ax.imshow(data, cmap='gray')
    ax.set_title("Noise Pattern")
    ax.axis('off')

    ax = axes[1]
    ax.imshow(np.log1p(magnitude), cmap='gray')
    ax.set_title("FFT Magnitude (log)")
    ax.axis('off')

    ax = axes[2]
    ax.imshow(np.log1p(power_spectrum), cmap='gray')
    ax.set_title("Power Spectrum (log)")
    ax.axis('off')

    ax = axes[3]
    power_norm = power_spectrum / np.max(power_spectrum)
    im = ax.imshow(power_norm, cmap='hot', vmin=0, vmax=0.1)

    h, w = power_norm.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    radial_profile = []
    for r in range(min(cx, cy)):
        mask = (R >= r) & (R < r + 1)
        if np.any(mask):
            radial_profile.append(np.mean(power_spectrum[mask]))
        else:
            radial_profile.append(0)

    peak_radius = np.argmax(radial_profile)

    ax.set_title(f"Band Structure\n(peak at r≈{peak_radius})")
    ax.axis('off')

    ax = axes[4]
    radii = np.arange(len(radial_profile))
    normalized_freq = radii / (2 * len(radial_profile))

    ax.semilogy(normalized_freq, radial_profile, 'b-', linewidth=2)
    ax.axvline(normalized_freq[peak_radius], color='r', linestyle='--', 
               label=f'Peak at f≈{normalized_freq[peak_radius]:.3f}')
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Power')
    ax.set_title('Radial Power Spectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 0.5)

    plt.tight_layout()
    return fig, peak_radius

def analyze_files(two_d_files, three_d_files):
    """分析给定的2D和3D噪声文件列表
    
    Args:
        two_d_files: 2D噪声文件列表
        three_d_files: 3D切片噪声文件列表
    """
    print("Visualizing Wavelet Noise Band-Limited Property...")
    print("=" * 60)
    
    # 生成單張分析圖
    for filename in two_d_files:
        octave = filename.split('_')[-1].replace('.raw', '')
        title = f"Pure 2D - Octave {octave}"
        fig, peak_r = analyze_single_band(filename, title)
        if fig:
            plt.savefig(f'{filename.replace(".raw", "")}_analysis.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()
    
    for filename in three_d_files:
        octave = filename.split('_')[-1].replace('.raw', '')
        title = f"3D Sliced - Octave {octave}"
        fig, peak_r = analyze_single_band(filename, title)
        if fig:
            plt.savefig(f'{filename.replace(".raw", "")}_analysis.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()
    
    # 生成比較圖
    create_comparison_figure(two_d_files, three_d_files)
    
    print("\nAnalysis complete!")
    print("The comparison figure 'wavelet_2D_vs_3D_comparison.png' has been generated.")
    print("Individual analysis figures have been saved for each octave.")
    print("Observe the clear band separation (dark center) in the Pure 2D power spectra,")
    print("versus the low-frequency leakage in the 3D Sliced power spectra.")

if __name__ == "__main__":
    two_d_files = [f"wavelet_noise_2D_octave_{oct}.raw" for oct in [5, 6, 7]]
    three_d_files = [f"wavelet_noise_3Dsliced_octave_{oct}.raw" for oct in [3, 4, 5]]
    
    analyze_files(two_d_files, three_d_files) 