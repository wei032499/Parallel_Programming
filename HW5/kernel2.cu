#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int pitch, int *output, int resX, int resY, int maxIterations) {

    int thisX = threadIdx.x + blockIdx.x * blockDim.x;
    int thisY = threadIdx.y + blockIdx.y * blockDim.y;

    if(thisX >= resX || thisY >= resY)
        return;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    int index = thisX + thisY * (pitch/sizeof(int));
    output[index] = mandel(x, y, maxIterations);

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *output;
    int *outputD;
    size_t pitch;

   
    // allocate page-locked and mapped memory
    cudaError_t err = cudaHostAlloc(&output, resX * resY * sizeof(int), cudaHostAllocMapped);
    // printf("%s\n",cudaGetErrorString(err));
    
    cudaMallocPitch(&outputD, &pitch, resX * sizeof(int), resY);
    

    int blockDimX = 16;
    int blockDimY = 16;

    dim3 dimBlock(blockDimX, blockDimY); // # threads per block
    dim3 dimGrid((int)ceil((double)resX/blockDimX), (int)ceil((double)resY/blockDimY)); // # blocks per grid

    mandelKernel<<<dimGrid, dimBlock>>>(stepX, stepY, lowerX, lowerY, pitch, outputD, resX, resY, maxIterations);

    cudaMemcpy2D(output, resX * sizeof(int), outputD, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, output, resX * resY * sizeof(int));
    
    cudaFreeHost(output);
    cudaFree(outputD);
}
