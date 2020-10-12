CXX = nvcc 
CXX_FLAGS = 

main: 
	$(CXX) main.cpp -o rtc $(CXX_FLAGS)

render:
	./rtc
	convert output.ppm output.png

clean: 
	rm rtc
	rm output.ppm
	rm output.png
