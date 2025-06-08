CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter
INCLUDES = -I.

# 預設目標
all: test_noise main

# 編譯噪聲測試程序
test_noise: test_noise.cpp noise_utils.cpp perlin.h vec3.h rtweekend.h stb_image_write.h noise_utils.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o test_noise test_noise.cpp noise_utils.cpp

# 編譯主程序（光線追踪器）
main: main.cpp perlin.h vec3.h rtweekend.h stb_image_write.h texture.h material.h sphere.h hittable.h hittable_list.h quad.h ray.h interval.h color.h aabb.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main main.cpp

# 執行噪聲測試程序
run_test: test_noise
	./test_noise

# 執行主程序
run_main: main
	./main

# 清理生成的檔案
clean:
	rm -f test_noise main
	rm -f *.png *.ppm

# 清理所有輸出圖像
clean_images:
	rm -f *.png *.ppm

# 幫助資訊
help:
	@echo "可用目標:"
	@echo "  all              - 編譯所有程式"
	@echo "  test_noise       - 編譯噪聲測試程序"
	@echo "  main             - 編譯主程序（光線追踪器）"
	@echo "  run_test         - 執行噪聲測試程序"
	@echo "  run_main         - 執行主程序"
	@echo "  clean            - 清理執行檔"
	@echo "  clean_images     - 清理所有圖像檔案"
	@echo "  help             - 顯示此幫助資訊"

.PHONY: all run_test run_main clean clean_images help 