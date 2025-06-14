CXX = g++
CXXFLAGS = -Wall -std=c++11

TARGET = wavelet_noise
SRCS = main.cpp WaveletNoise.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
	rm -f *.raw
	rm -f *.png

help:
	@echo "可用的命令："
	@echo "  make        - 編譯 wavelet_noise"
	@echo "  make run    - 執行 wavelet_noise"
	@echo "  make clean  - 清理生成的文件"

.PHONY: all clean run help 