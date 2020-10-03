CXX = g++
CXX_FLAGS = 

main: 
	$(CXX) main.cpp -o rt $(CXX_FLAGS)

render:
	./rt
	convert output.ppm output.png

clean: 
	rm rt
	rm output.ppm
	rm output.png
