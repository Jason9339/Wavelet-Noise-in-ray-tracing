# Makefile for Wavelet Noise Project
# 編譯器設定
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native
LDFLAGS = -lm

# 目標檔案設定
TARGET = wavelet_test
SOURCE = wavelet_test.cpp
OBJECTS = $(SOURCE:.cpp=.o)

# 預設目標
all: $(TARGET)

# 主要可執行檔編譯規則
$(TARGET): $(OBJECTS)
	@echo "正在連結 $(TARGET)..."
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "編譯完成！"

# 物件檔編譯規則
%.o: %.cpp
	@echo "正在編譯 $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 執行程式
run: $(TARGET)
	@echo "執行 $(TARGET)..."
	./$(TARGET)

# 清理產生的檔案
clean:
	@echo "清理中..."
	rm -f $(OBJECTS) $(TARGET)
	rm -f *.bmp *.csv
	@echo "清理完成！"

# 清理所有產生的檔案（包括除錯和診斷檔案）
clean-all: clean
	@echo "清理所有產生的檔案..."
	rm -f debug_*.bmp debug_*.csv
	rm -f diagnostic_*.bmp diagnostic_*.csv
	@echo "完全清理完成！"

# 除錯版本編譯
debug: CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0 -DDEBUG
debug: $(TARGET)
	@echo "除錯版本編譯完成！"

# 分析產生的檔案（執行 Python 分析腳本）
analyze: run
	@echo "執行 Python 分析腳本..."
	@if [ -f "analyzer.ipynb" ]; then \
		echo "請手動執行 Jupyter Notebook: analyzer.ipynb"; \
	else \
		echo "找不到 analyzer.ipynb 檔案"; \
	fi

# 檢查編譯器版本和功能
check:
	@echo "檢查編譯環境..."
	@echo "C++ 編譯器: $(CXX)"
	$(CXX) --version
	@echo "編譯器標記: $(CXXFLAGS)"

# 安裝依賴（如果需要的話）
install-deps:
	@echo "檢查系統依賴..."
	@echo "此專案僅需要標準 C++ 程式庫和數學程式庫"
	@echo "確保已安裝 g++ 或 clang++"

# 幫助資訊
help:
	@echo "可用的 make 目標："
	@echo "  all       - 編譯專案（預設）"
	@echo "  run       - 編譯並執行程式"
	@echo "  debug     - 編譯除錯版本"
	@echo "  clean     - 清理編譯產生的檔案"
	@echo "  clean-all - 清理所有產生的檔案"
	@echo "  analyze   - 執行程式並提示分析結果"
	@echo "  check     - 檢查編譯環境"
	@echo "  help      - 顯示此幫助資訊"

# 標記為偽目標（不是實際檔案）
.PHONY: all run clean clean-all debug analyze check install-deps help 