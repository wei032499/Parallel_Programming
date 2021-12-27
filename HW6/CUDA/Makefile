default: conv

CXX=g++
CXXFLAGS= -O3 -std=c++17 -Wall -g -fPIC -lm -w -g

OBJS = main.o bmpfuncs.o serialConv.o helper.o kernel.o

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

conv: $(OBJS)
	nvcc ${CUDA_LINK_FLAGS} -o $@ $(OBJS)

%.o: %.c
	$(CXX) -c $(CXXFLAGS) $< -o $@

kernel.o : kernel.cu kernel.h
	nvcc ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp
