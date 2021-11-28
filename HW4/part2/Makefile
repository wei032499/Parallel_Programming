all: matmul

matmul: main.cc matmul.cpp
	mpicxx -O3 main.cc matmul.cpp -o matmul

clean:
	rm -f matmul