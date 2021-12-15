#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

/** block size along */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32


/** a useful function to compute the number of threads */
__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__device__ int same_dwell(int d1, int d2) {
	if(d1 == d2)
		return d1;
	else if(d1 == -2 || d2 == -2) // -2 is uncomputed
		return max(d1, d2);
	else
		return -1; // different
}

__device__ int border_dwell(float stepX, float stepY, float lowerX, float lowerY, int x0, int y0, int d, int maxIterations) {
	// check whether all boundary pixels have the same dwell
	int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread id
	int bs = blockDim.x * blockDim.y; // block size
	int comm_dwell = -2;
	// for all boundary pixels, distributed across threads
	for(int r = tid; r < d; r += bs) {
		// for each boundary: b = 0, 1, 2, 3 : right, bottom, left, top
		for(int b = 0; b < 4; b++) {
			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
			// int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
      float c_re = lowerX + x * stepX;
      float c_im = lowerY + y * stepY;
      int dwell = mandel(c_re, c_im, maxIterations);
			comm_dwell = same_dwell(comm_dwell, dwell);
		}
	}  // for all boundary pixels
	// reduce across threads in the block
	__shared__ int ldwells[BSX * BSY];
	int nt = min(d, BSX * BSY);
	if(tid < nt)
		ldwells[tid] = comm_dwell;
	__syncthreads();
	for(; nt > 1; nt /= 2) {
		if(tid < nt / 2)
			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
		__syncthreads();
	}
	return ldwells[0];
}

/** the kernel to fill the image region with a specific dwell value */
__global__ void dwell_fill_k(int *output, int resX, int resY, int x0, int y0, int d, int dwell) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < d && y < d) {
		x += x0, y += y0;
    if(x >= resX || y >= resY)
      return;
		output[y * resX + x] = dwell;
	}
}

/* the kernel to fill in per-pixel values of the portion of the Mandelbrot set */
__global__ void mandelbrot_pixel_k(int *output, float stepX, float stepY, float lowerX, float lowerY, int resX, int resY, int x0, int y0, int d, int maxIterations) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < d && y < d) {
		x += x0, y += y0;
    if(x >= resX || y >= resY)
      return;
    float c_re = lowerX + x * stepX;
    float c_im = lowerY + y * stepY;
		output[y * resX + x] = mandel(c_re, c_im, maxIterations);
	}
}


__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int resX, int resY, int x0, int y0, int d, int *output, int maxIterations, int depth) {
  
  x0 += d * blockIdx.x, y0 += d * blockIdx.y;
	int comm_dwell = border_dwell(stepX, stepY, lowerX, lowerY, x0, y0, d, maxIterations);
	if(threadIdx.x == 0 && threadIdx.y == 0) {
		if(comm_dwell != -1) {
			// uniform dwell, just fill
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			dwell_fill_k<<<grid, bs>>>(output, resX, resY, x0, y0, d, comm_dwell);
		} else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
			// subdivide recursively
			dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
			mandelKernel<<<grid, bs>>>(stepX, stepY, lowerX, lowerY, resX, resY, x0, y0, d / SUBDIV, output, maxIterations, depth	+ 1);
		} else {
			// leaf, per-pixel kernel
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			mandelbrot_pixel_k<<<grid, bs>>>(output, stepX, stepY, lowerX, lowerY, resX, resY, x0, y0, d, maxIterations);
		}
	}

}


// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // int *optput;
    // cudaMalloc((void**)&optput, sizeof(int) * resX * resY);


    
    cudaHostRegister( img, resX * resY * sizeof(int), cudaHostRegisterPortable );

  
    dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
	  mandelKernel<<<grid, bs>>>(stepX, stepY, lowerX, lowerY, resX, resY, 0, 0, resX / INIT_SUBDIV, img, maxIterations, 1);
    
    // printf("%s\n",cudaGetErrorString(cudaGetLastError()));

    cudaHostUnregister(img);

}