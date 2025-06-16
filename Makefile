CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter
INCLUDES = -I.

WAVELET_OBJ = WaveletNoise.o

all: main

WaveletNoise.o: WaveletNoise.cpp WaveletNoise.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c WaveletNoise.cpp -o WaveletNoise.o

main: main.cpp $(WAVELET_OBJ) perlin.h vec3.h rtweekend.h stb_image_write.h texture.h material.h sphere.h hittable.h hittable_list.h quad.h ray.h interval.h color.h aabb.h WaveletNoise.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o main main.cpp $(WAVELET_OBJ)

run: main
	./main

clean:
	rm -f main *.o *.png *.ppm
	rm -f result/*.png result/*.ppm
	rm -f result_raytracing/*.png result_raytracing/*.ppm

clean_images:
	rm -f *.png *.ppm
	rm -f result_raytracing/*.png result_raytracing/*.ppm

compare: main
	@echo "生成 Perlin noise 渲染..."
	echo "0" | ./main
	@echo "生成 Wavelet 3D noise 渲染..."
	echo -e "1\n4" | ./main
	@echo "所有渲染完成"

help:
	@echo "可用目標:"
	@echo "  all       - 編譯程序"
	@echo "  run       - 執行程序"
	@echo "  compare   - 生成所有噪聲類型比較"
	@echo "  clean     - 清理執行檔和圖像"
	@echo "  help      - 顯示幫助"

.PHONY: all run clean clean_images compare help 