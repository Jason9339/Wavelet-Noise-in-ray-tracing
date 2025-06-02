CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
INCLUDES = -I.

# 預設目標
all: perlin_viewer perlin_interactive

# 編譯自動化生成器
perlin_viewer: perlin_viewer.cpp perlin.h vec3.h rtweekend.h stb_image_write.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o perlin_viewer perlin_viewer.cpp

# 編譯互動式生成器
perlin_interactive: perlin_interactive.cpp perlin.h vec3.h rtweekend.h stb_image_write.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o perlin_interactive perlin_interactive.cpp

# 執行自動化生成器
run_viewer: perlin_viewer
	./perlin_viewer

# 執行互動式生成器
run_interactive: perlin_interactive
	./perlin_interactive

# 清理生成的檔案
clean:
	rm -f perlin_viewer perlin_interactive
	rm -f *.png *.ppm

# 清理所有輸出圖像
clean_images:
	rm -f *.png *.ppm

# 幫助資訊
help:
	@echo "可用目標:"
	@echo "  all              - 編譯所有程式"
	@echo "  perlin_viewer    - 編譯自動化生成器"
	@echo "  perlin_interactive - 編譯互動式生成器"
	@echo "  run_viewer       - 執行自動化生成器"
	@echo "  run_interactive  - 執行互動式生成器"
	@echo "  clean            - 清理執行檔"
	@echo "  clean_images     - 清理所有圖像檔案"
	@echo "  help             - 顯示此幫助資訊"

.PHONY: all run_viewer run_interactive clean clean_images help 