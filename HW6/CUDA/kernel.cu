#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, unsigned char *inputImage, float *outputImage) {

    int half_filterWidth = filterWidth / 2;

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int x = index % imageWidth;
    int y = index / imageWidth;

    float sum = 0.0;

    int row_start = x - half_filterWidth >=0 ? 0 : half_filterWidth - x;
    int row_end = x + half_filterWidth < imageWidth ? filterWidth-1 : imageWidth-x;

    int col_start = y - half_filterWidth >=0 ? 0 : half_filterWidth - y;
    int col_end = y + half_filterWidth < imageHeight ? filterWidth-1 : imageHeight-y;


    
    if(row_end-row_start+1==filterWidth && col_end-col_start+1==filterWidth)
    {
        if(filterWidth == 3)
        #pragma unroll
        for (int row = 0; row < 3; row++)
        {
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 0)] * filter[row * filterWidth + 0];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 1)] * filter[row * filterWidth + 1];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 2)] * filter[row * filterWidth + 2];
        }
        else if(filterWidth == 5)
        #pragma unroll
        for (int row = 0; row < 5; row++)
        {
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 0)] * filter[row * filterWidth + 0];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 1)] * filter[row * filterWidth + 1];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 2)] * filter[row * filterWidth + 2];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 3)] * filter[row * filterWidth + 3];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 4)] * filter[row * filterWidth + 4];
        }
        else if(filterWidth == 7)
        #pragma unroll
        for (int row = 0; row < 7; row++)
        {
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 0)] * filter[row * filterWidth + 0];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 1)] * filter[row * filterWidth + 1];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 2)] * filter[row * filterWidth + 2];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 3)] * filter[row * filterWidth + 3];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 4)] * filter[row * filterWidth + 4];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 5)] * filter[row * filterWidth + 5];
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + 6)] * filter[row * filterWidth + 6];
        }
        else
        for (int row = row_start; row <= row_end; row++)
        {
            for (int col = col_start; col <= col_end; col++)
            {
                sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + col)] * filter[row * filterWidth + col];
            }
        }
    }
    else
    for (int row = row_start; row <= row_end; row++)
    {
        for (int col = col_start; col <= col_end; col++)
        {
            sum += inputImage[(y - half_filterWidth + row) * imageWidth + (x - half_filterWidth + col)] * filter[row * filterWidth + col];
        }
    }
    outputImage[y * imageWidth + x] = sum;

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage)
{
    float *filterO = filter;
    unsigned char *inputImageU = (unsigned char *)malloc(sizeof(unsigned char) * imageHeight*imageWidth);
    for(int i = 0;i<imageHeight * imageWidth;i++)
        inputImageU[i] = (unsigned char)inputImage[i];
    int padding = 0;
    for(int i=0;i<filterWidth/2;i++)
    {
        int isZero = 1;
        for(int j=i;j<filterWidth-i;j++)
            if( filter[i*filterWidth + j] !=0 ||
                filter[(filterWidth-i-1)*filterWidth + j] !=0 ||
                filter[j*filterWidth + i] !=0 ||
                filter[j*filterWidth + (filterWidth-i-1)] !=0
            ){
                isZero = 0;
                break;
            }
        
        if(isZero==0)
            break;
        else
            padding++;

    }
    padding *= 2;
    int newWidth = filterWidth-padding;
    float *filterM = (float *)malloc(sizeof(float)*newWidth*newWidth);
    for(int i=0;i<newWidth;i++)
        memcpy(filterM+i*newWidth, filter+(i+padding/2)*filterWidth+padding/2, sizeof(float)*newWidth);
    filter = filterM;
    filterWidth = newWidth;

    int filterSize = filterWidth * filterWidth;


    // cudaHostRegister( filter, filterSize * sizeof(float), cudaHostRegisterPortable );
    // cudaHostRegister( inputImageU, imageHeight * imageWidth * sizeof(unsigned char), cudaHostRegisterPortable );
    // cudaHostRegister( outputImage, imageHeight * imageWidth * sizeof(float), cudaHostRegisterPortable );
  
    float *filterD = nullptr;
    unsigned char *inputImageUD = nullptr;
    float *outputImageD = nullptr;

    cudaMalloc(&filterD, filterSize * sizeof(float));
    cudaMemcpy(filterD, filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&inputImageUD, imageHeight * imageWidth * sizeof(unsigned char));
    cudaMemcpy(inputImageUD, inputImageU, imageHeight * imageWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&outputImageD, imageHeight * imageWidth * sizeof(float));


    // cudaHostGetDevicePointer((void **)&filterD, (void *)filter, cudaHostRegisterDefault);
    // cudaHostGetDevicePointer((void **)&inputImageUD, (void *)inputImageU, cudaHostRegisterDefault);
    // cudaHostGetDevicePointer((void **)&outputImageD, (void *)outputImage, cudaHostRegisterDefault);
    
    convolution<<<imageHeight*imageWidth/256, 256>>>(filterWidth, filterD, imageHeight, imageWidth, inputImageUD, outputImageD);
    // printf("%s\n",cudaGetErrorString(cudaGetLastError()));

    cudaMemcpy(outputImage, outputImageD, imageHeight * imageWidth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(filterD);
    cudaFree(inputImageUD);
    cudaFree(outputImageD);
    // cudaHostUnregister(outputImage);

    free(inputImageU);
    free(filter);
    filter = filterO;

}