CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter
INCLUDES = -I.

# 預設目標
all: main

# 編譯主程序（光線追踪器）
main: main.cpp perlin.h vec3.h rtweekend.h stb_image_write.h texture.h material.h sphere.h hittable.h hittable_list.h quad.h ray.h interval.h color.h aabb.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main main.cpp

# 執行主程序
run: main
	./main

# 清理生成的檔案
clean:
	rm -f main
	rm -f *.png *.ppm
	rm -f result/*.png result/*.ppm

# 清理所有輸出圖像
clean_images:
	rm -f *.png *.ppm

# 幫助資訊
help:
	@echo "可用目標:"
	@echo "  all              - 編譯光線追踪程序"
	@echo "  main             - 編譯主程序（光線追踪器）"
	@echo "  run              - 執行光線追踪程序"
	@echo "  clean            - 清理執行檔和圖像"
	@echo "  clean_images     - 清理所有圖像檔案"
	@echo "  help             - 顯示此幫助資訊"

.PHONY: all run clean clean_images help 