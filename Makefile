CXX = g++ 
CXX_FLAGS = 
NVXX = nvcc

.PHONY: main cuda render clean

main:
	$(CXX) cpu/main.cpp -o rt $(CXX_FLAGS)

cuda: 
	$(NVXX) cuda/main.cu -o rt

render:
	./rt
	convert output.ppm output.png

clean: 
	rm rt
	rm output.ppm
	rm output.png
