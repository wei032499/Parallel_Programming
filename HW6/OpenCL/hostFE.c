#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
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

    cl_int status, myqueue_err, filter_err, output_err, input_err, kernel_err;

    cl_command_queue myqueue;
    myqueue = clCreateCommandQueue(*context, device[0], 0, &myqueue_err);

    cl_mem myfilter = clCreateBuffer(*context, 0, sizeof(float)*filterSize, NULL, &filter_err);

    cl_mem input = clCreateBuffer(*context, 0, sizeof(unsigned char)*imageHeight*imageWidth, NULL, &input_err);

    cl_mem output = clCreateBuffer(*context, 0, sizeof(float)*imageHeight*imageWidth, NULL, &output_err);


    clEnqueueWriteBuffer(myqueue, input, CL_MEM_READ_ONLY, 0, imageHeight*imageWidth * sizeof(unsigned char), inputImageU, 0, NULL, NULL);
    clEnqueueWriteBuffer(myqueue, myfilter, CL_MEM_READ_ONLY, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(*program, "convolution", &kernel_err);
    clSetKernelArg(kernel, 0, sizeof(int), &filterWidth );
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &myfilter);
    clSetKernelArg(kernel, 2, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);

    size_t globalSize[2] = { imageWidth, imageHeight };
    clEnqueueNDRangeKernel(myqueue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);


    clEnqueueReadBuffer(myqueue, output, CL_TRUE, 0, sizeof(float) * imageHeight*imageWidth, outputImage, 0, NULL, NULL);

    clReleaseMemObject(output);
    clReleaseMemObject(input);
    clReleaseMemObject(myqueue);
    clReleaseMemObject(kernel);

    free(inputImageU);
    free(filter);
    filter = filterO;

}