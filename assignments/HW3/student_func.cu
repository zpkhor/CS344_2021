/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

unsigned int **d_histo_arr = nullptr;

__global__ void local_histo(const float *const lum,
                           unsigned int **const d_histo_arr,
                           float lumMin,
                           float lumRange,
                           const size_t n_elements,
                           const size_t numBins) {
   extern __shared__ unsigned int histo[];
   unsigned int arr_i = blockIdx.x;
   unsigned int global_i = arr_i * 16;
   // Init histo
   for (int j = 0; j < numBins; j++) {
      histo[j] = 0;
   }

   unsigned int *cur_arr = d_histo_arr[arr_i];
   unsigned int bound = 16 + global_i;
   unsigned int max_bin = numBins - 1;
   while (global_i < n_elements && global_i < bound) {
      unsigned int bin = (unsigned int) ((lum[global_i] - lumMin) / lumRange * numBins);
      if (bin > max_bin) {
         bin = max_bin;
      }
      histo[bin]++;
      global_i++;
   }

   for (int k = 0; k < numBins; k++) {
      cur_arr[k] = histo[k];
   }
}

__global__ void reduce_kernel(unsigned int **d_histo_arr_out, unsigned int **d_histo_arr_in, const int num_arr, const int numBins)
{
   unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
   int tid  = threadIdx.x;

   // do reduction in global mem
   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if (tid < s && myId < num_arr && (myId + s) < num_arr)
      {
         unsigned int *d_histo_in = d_histo_arr_in[myId];
         unsigned int *d_histo_in_s = d_histo_arr_in[myId + s];
         for (int i=0; i<numBins;++i) {
            d_histo_in[i] += d_histo_in_s[i];
         }
      }
      __syncthreads();        // make sure all adds at one stage are done!
   }

   // only thread 0 writes result for this block back to global mem
   if (tid == 0 && myId < num_arr)
   {
      unsigned int *d_histo_out = d_histo_arr_out[blockIdx.x];
      unsigned int *d_histo_reduced = d_histo_arr_in[myId];
      for (int i=0; i<numBins;++i) {
         d_histo_out[i] = d_histo_reduced[i];
      }
   }
}

__global__ void exclusive_scan(unsigned int *const d_cdf, unsigned int *d_reduced_histo) {
   int idx = threadIdx.x;
   if (idx > 0) {
      d_cdf[idx] = d_reduced_histo[idx - 1];
   } else {
      d_cdf[0] = 0;
   }
   __syncthreads();

   for (int s = 1; s < blockDim.x; s<<=1) {
      if ((idx - s) >= 0) {
         d_cdf[idx] += d_cdf[idx -s];
      }
      __syncthreads();
   }
}

int roundUpToPowerOf2(int v) {
    v--; // Ensure that v is not already a power of 2
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++; // Increment to get the next power of 2
    return v;
}

__global__ void minReduce(const float* const input, int size, float* result) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < size) ? input[i] : 0xFFFFFF;
    __syncthreads();

    // Perform parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Store result in global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void maxReduce(const float* const input, int size, float* result) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < size) ? input[i] : -0xFFFFFF;
    __syncthreads();

    // Perform parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Store result in global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

void cleanup(int num_arr, int** h_histo_arr) {
  for (int i=0; i<num_arr; i++) {
   checkCudaErrors(cudaFree(h_histo_arr[i]));
  }
  checkCudaErrors(cudaFree(d_histo_arr));
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf, float &min_logLum,
                                  float &max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
  // TODO
// Here are the steps you need to implement
   // 1) find the minimum and maximum value in the input logLuminance channel
   //    store in min_logLum and max_logLum
   unsigned int n_elements = numRows * numCols;
   const int blocksPerGrid = roundUpToPowerOf2((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
   float *d_logLum, *d_logLum_imt;
   cudaMalloc((void **) &d_logLum_imt, blocksPerGrid * sizeof(float));
   cudaMalloc((void **) &d_logLum, sizeof(float));
   // max
   maxReduce<<<blocksPerGrid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_logLuminance, n_elements, d_logLum_imt);
   checkCudaErrors(cudaDeviceSynchronize());
   maxReduce<<<1, blocksPerGrid, blocksPerGrid * sizeof(float)>>>(d_logLum_imt, blocksPerGrid, d_logLum);
   checkCudaErrors(cudaDeviceSynchronize());
   cudaMemcpy(&max_logLum, d_logLum, sizeof(float), cudaMemcpyDeviceToHost);
   // min
   minReduce<<<blocksPerGrid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_logLuminance, n_elements, d_logLum_imt);
   checkCudaErrors(cudaDeviceSynchronize());
   minReduce<<<1, blocksPerGrid, blocksPerGrid * sizeof(float)>>>(d_logLum_imt, blocksPerGrid, d_logLum);
   checkCudaErrors(cudaDeviceSynchronize());
   cudaMemcpy(&min_logLum, d_logLum, sizeof(float), cudaMemcpyDeviceToHost);
   cudaFree(d_logLum);
   cudaFree(d_logLum_imt);
   
   // 2) subtract them to find the range
   float lumRange = max_logLum - min_logLum;

   // 3) generate a histogram of all the values in the logLuminance channel using
   const dim3 blockSize(1);
   const dim3 gridSize((n_elements + 15) / 16);
   int num_arr = roundUpToPowerOf2(gridSize.x * gridSize.y);
   checkCudaErrors(cudaMalloc((void **) &d_histo_arr, num_arr * sizeof(unsigned int *)));
   int **h_histo_arr = (int**) malloc(num_arr * sizeof(int *));
   for (int i=0; i<num_arr; ++i) {
      checkCudaErrors(cudaMalloc(&h_histo_arr[i], numBins * sizeof(int)));
   }
   checkCudaErrors(cudaMemcpy(d_histo_arr, h_histo_arr, num_arr * sizeof(unsigned int*), cudaMemcpyHostToDevice));
   local_histo<<<gridSize, blockSize, numBins * sizeof(int)>>>(d_logLuminance, d_histo_arr, min_logLum, lumRange, n_elements, numBins);
   checkCudaErrors(cudaDeviceSynchronize());

   // reduce
   unsigned int **d_reduced_intermediate_histo = nullptr;
   unsigned int **h_intermediate = (unsigned int**) malloc(num_arr * sizeof(unsigned int *));
   for (int i=0; i<num_arr; ++i) {
      checkCudaErrors(cudaMalloc(&h_intermediate[i], numBins * sizeof(unsigned int)));
   }
   checkCudaErrors(cudaMalloc((void **) &d_reduced_intermediate_histo, num_arr * sizeof(unsigned int *)));
   checkCudaErrors(cudaMemcpy(d_reduced_intermediate_histo, h_intermediate, num_arr * sizeof(unsigned int*), cudaMemcpyHostToDevice));

   int n_blocks = (num_arr + 1023) / BLOCK_SIZE;
   reduce_kernel<<<n_blocks, BLOCK_SIZE>>>(d_reduced_intermediate_histo, d_histo_arr, num_arr, numBins);
   checkCudaErrors(cudaDeviceSynchronize());

   int n_threads = n_blocks;
   n_blocks = 1;
   unsigned int **d_reduced_histo = nullptr;
   unsigned int **h_reduced = (unsigned int**) malloc(1 * sizeof(unsigned int *));
   checkCudaErrors(cudaMalloc((void **) &d_reduced_histo, sizeof(unsigned int *)));
   checkCudaErrors(cudaMalloc(&h_reduced[0], numBins * sizeof(unsigned int)));
   cudaMemcpy(d_reduced_histo, h_reduced, 1 * sizeof(unsigned int*), cudaMemcpyHostToDevice);
   reduce_kernel<<<n_blocks, n_threads>>>(d_reduced_histo, d_reduced_intermediate_histo, n_threads, numBins);
   checkCudaErrors(cudaDeviceSynchronize());

   // 4) Perform an exclusive scan (prefix sum) on the histogram to get
   //    the cumulative distribution of luminance values (this should go in the
   //    incoming d_cdf pointer which already has been allocated for you)

   exclusive_scan<<<1, numBins>>>(d_cdf, *h_reduced);
   checkCudaErrors(cudaDeviceSynchronize());
   cleanup(num_arr, h_histo_arr);
   free(h_histo_arr);
}

/* Lesson learned 2024.01.01, took me 3 days 2 nights
1. Always checkCudaError after kernel call, malloc, memcopy
2. Array of pointers allocation need to be done on host first, then copy to ** on device,
 as the array of pointers point to device memory:
   unsigned int **d_reduced_intermediate_histo = nullptr;
   unsigned int **h_intermediate = (unsigned int**) malloc(num_arr * sizeof(unsigned int *));
   for (int i=0; i<num_arr; ++i) {
      checkCudaErrors(cudaMalloc(&h_intermediate[i], numBins * sizeof(unsigned int)));
   }
   checkCudaErrors(cudaMalloc((void **) &d_reduced_intermediate_histo, num_arr * sizeof(unsigned int *)));
   checkCudaErrors(cudaMemcpy(d_reduced_intermediate_histo, h_intermediate, num_arr * sizeof(unsigned int*), cudaMemcpyHostToDevice));
3. Need to pad the input to the size of next round up of power of 2
4. When casting to int, ensure bracket is placed entirely on the float calculation
*/