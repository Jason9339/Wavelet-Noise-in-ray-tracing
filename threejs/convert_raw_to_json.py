#!/usr/bin/env python3
"""
通用的 RAW 到 JSON 轉換工具
支持將二進制 float 文件轉換為 JSON 格式，方便 Web 應用讀取
"""
import struct
import json
import numpy as np
import os
import argparse

def convert_raw_to_json(raw_file_path, json_file_path, image_size=256):
    """
    將二進制 float raw 文件轉換為 JSON 格式
    
    Args:
        raw_file_path: 輸入的 .raw 文件路徑
        json_file_path: 輸出的 .json 文件路徑
        image_size: 圖像尺寸 (預設 256x256)
    """
    try:
        # 檢查輸入文件是否存在
        if not os.path.exists(raw_file_path):
            print(f"錯誤: 文件不存在 - {raw_file_path}")
            return False
        
        # 讀取二進制 float 數據
        with open(raw_file_path, 'rb') as f:
            data = f.read()
        
        # 解析為 float 陣列
        float_count = len(data) // 4
        floats = struct.unpack(f'{float_count}f', data)
        
        # 驗證數據大小
        expected_size = image_size * image_size
        if float_count != expected_size:
            print(f"警告: 數據大小不匹配 - 期望 {expected_size}，實際 {float_count}")
            image_size = int(np.sqrt(float_count))
            print(f"自動調整圖像大小為: {image_size}x{image_size}")
        
        # 轉換為 numpy 陣列並重塑為 2D
        array = np.array(floats).reshape(image_size, image_size)
        
        # 計算統計信息
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        mean_val = float(np.mean(array))
        std_val = float(np.std(array))
        
        # 歸一化到 [0, 1]
        if max_val != min_val:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(array)
            print("警告: 數據值全部相同，歸一化為 0")
        
        # 準備 JSON 數據
        json_data = {
            "width": image_size,
            "height": image_size,
            "original_range": {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val
            },
            "data": normalized.flatten().tolist()
        }
        
        # 創建輸出目錄
        output_dir = os.path.dirname(json_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 寫入 JSON 文件
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, separators=(',', ':'))
        
        print(f"轉換完成: {raw_file_path} -> {json_file_path}")
        print(f"   圖像尺寸: {image_size}x{image_size}")
        print(f"   數據範圍: [{min_val:.4f}, {max_val:.4f}]")
        print(f"   平均值: {mean_val:.4f}, 標準差: {std_val:.4f}")
        print(f"   JSON 文件大小: {len(json.dumps(json_data))} bytes")
        
        return True
        
    except Exception as e:
        print(f"轉換失敗: {e}")
        return False

def batch_convert(octaves=None, noise_types=None):
    """
    批量轉換 experient 目錄中的噪聲數據
    
    Args:
        octaves: 要轉換的 octave 列表，預設為 [3, 4, 5]
        noise_types: 要轉換的噪聲類型，預設為所有類型
    """
    base_dir = "../experient/result_raw"
    output_dir = "result_json"
    
    # 預設轉換所有 octave
    if octaves is None:
        octaves = [3, 4, 5]
    
    # 預設轉換所有噪聲類型
    if noise_types is None:
        noise_types = ['wavelet', 'perlin']
    
    # 生成要轉換的文件列表
    files_to_convert = []
    
    for octave in octaves:
        for noise_type in noise_types:
            if noise_type == 'wavelet':
                files_to_convert.extend([
                    (f"wavelet_noise_2D_octave_{octave}.raw", 
                     f"wavelet_noise_2d_octave{octave}.json"),
                    (f"wavelet_noise_3Dsliced_octave_{octave}.raw", 
                     f"wavelet_noise_3d_sliced_octave{octave}.json"),
                    (f"wavelet_noise_3Dprojected_octave_{octave}.raw", 
                     f"wavelet_noise_3d_projected_octave{octave}.json"),
                ])
            elif noise_type == 'perlin':
                files_to_convert.extend([
                    (f"perlin_noise_2D_octave_{octave}.raw", 
                     f"perlin_noise_2d_octave{octave}.json"),
                    (f"perlin_noise_3Dsliced_octave_{octave}.raw", 
                     f"perlin_noise_3d_sliced_octave{octave}.json"),
                ])
    
    success_count = 0
    total_count = len(files_to_convert)
    
    print("開始批量轉換...")
    print(f"來源目錄: {base_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"Octave: {octaves}")
    print(f"噪聲類型: {noise_types}")
    print(f"總計 {total_count} 個文件")
    print("-" * 60)
    
    for raw_file, json_file in files_to_convert:
        raw_path = os.path.join(base_dir, raw_file)
        json_path = os.path.join(output_dir, json_file)
        
        if convert_raw_to_json(raw_path, json_path):
            success_count += 1
        print("-" * 60)
    
    print(f"批量轉換完成！成功 {success_count}/{total_count} 個文件")
    
    print("\n轉換摘要:")
    for octave in octaves:
        octave_files = [f for f in files_to_convert if f"octave{octave}" in f[1]]
        print(f"  Octave {octave}: {len(octave_files)} 個文件")

def batch_convert_octave4_only():
    """僅轉換 octave4 數據"""
    return batch_convert(octaves=[4])

def batch_convert_specific_octave(octave):
    """轉換指定 octave 的所有數據"""
    return batch_convert(octaves=[octave])

def main():
    parser = argparse.ArgumentParser(description='將 RAW 文件轉換為 JSON 格式')
    parser.add_argument('--input', '-i', help='輸入的 RAW 文件路徑')
    parser.add_argument('--output', '-o', help='輸出的 JSON 文件路徑')
    parser.add_argument('--size', '-s', type=int, default=256, help='圖像尺寸 (預設: 256)')
    parser.add_argument('--batch', '-b', action='store_true', help='批量轉換所有數據')
    parser.add_argument('--octave', type=int, choices=[3, 4, 5], help='僅轉換指定 octave')
    parser.add_argument('--octaves', nargs='+', type=int, choices=[3, 4, 5], 
                       help='轉換多個指定 octave，例如: --octaves 3 4')
    parser.add_argument('--noise-type', choices=['wavelet', 'perlin'], 
                       help='僅轉換指定噪聲類型')
    
    args = parser.parse_args()
    
    if args.batch:
        # 根據參數決定轉換範圍
        octaves = None
        noise_types = None
        
        if args.octave:
            octaves = [args.octave]
        elif args.octaves:
            octaves = args.octaves
            
        if args.noise_type:
            noise_types = [args.noise_type]
            
        batch_convert(octaves, noise_types)
    elif args.input and args.output:
        convert_raw_to_json(args.input, args.output, args.size)
    else:
        print("RAW 到 JSON 轉換工具")
        print("=" * 50)
        print("使用方式:")
        print("  單個文件轉換:")
        print("    python convert_raw_to_json.py -i input.raw -o output.json")
        print("")
        print("  批量轉換:")
        print("    python convert_raw_to_json.py --batch                    # 轉換所有數據")
        print("    python convert_raw_to_json.py --batch --octave 4         # 僅轉換 octave 4")
        print("    python convert_raw_to_json.py --batch --octaves 3 5      # 轉換 octave 3 和 5")
        print("    python convert_raw_to_json.py --batch --noise-type wavelet  # 僅轉換 wavelet")
        print("")
        print("  組合使用:")
        print("    python convert_raw_to_json.py --batch --octave 4 --noise-type perlin")
        print("")
        print("  其他選項:")
        print("    --size 512                                        # 指定圖像尺寸")

if __name__ == "__main__":
    main() 