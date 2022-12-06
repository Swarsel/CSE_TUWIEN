# requirements on ubuntu/linux mint
# sudo apt-get install build-essential libopenblas-base libopenblas-dev libeigen3-dev

CXX=g++
CXXFLAGS=-std=c++11 -Wall -pedantic -O3 -ffast-math -march=native  
# when using the Eigen headers:
# CXXFLAGS=-std=c++11 -Wall -pedantic -O3 -ffast-math -DEIGEN_DONT_PARALLELIZE -march=native   

.DEFAULT_GOAL := all

all: benchmark 
 
benchmark: benchmark_MMM.cpp Makefile
	$(CXX) benchmark_MMM.cpp -o benchmark -v  $(CXXFLAGS) 
	# when linking to openblas:
	# $(CXX) benchmark_MMM.cpp -o benchmark -v -lopenblas $(CXXFLAGS) 

.PHONY: clean
clean:
	rm benchmark
