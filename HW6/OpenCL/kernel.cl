__kernel void convolution(int filterWidth, __constant float *filter, int imageHeight, int imageWidth, __global const unsigned char *inputImage, __global float *outputImage) 
{
    int half_filterWidth = filterWidth / 2;

    int x = get_global_id(0);
    int y = get_global_id(1);

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
