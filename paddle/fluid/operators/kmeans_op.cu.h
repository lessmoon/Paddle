/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template<typename T>
__global__
void kernel_gpu_colwise_div(const int rows, const int cols, const int channels, 
        const T* input_a, const T* input_b, T* output) {
    //row*col*channels
    CUDA_KERNEL_LOOP(i, rows*channels*cols) {
        output[i] = input_a[i]/input_b[i/cols];
    }
}

template<typename T>
void colwise_div(cudaStream_t stream, const int rows, const int cols, const int channels,
        const T* input_a, const T* input_b, T* output) {
    kernel_gpu_colwise_div<<<GET_BLOCKS(rows*cols*channels), CUDA_NUM_THREADS, 0, stream>>>(rows, cols,
            channels, input_a, input_b, output);
}

template<typename T>
__device__ 
void _swap_cuda(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
}

template<typename T>
//insert
__device__
void insert_sort(const T* input, const int k, int* index_helper) {
    //from end search
    for (int i = k - 1; i >= 0; i--) {
        if (input[index_helper[i]] < input[index_helper[i + 1]]) {
            _swap_cuda(index_helper[i], index_helper[i + 1]);
        } else {
            return;
        }
    }
}

template<typename T>
__global__
void kernel_channelwise_topk_filter(const T* input, T* output, const int k, const int rows, const int cols,  const int channels,
        int* index_helper) {
    CUDA_KERNEL_LOOP(i, rows*channels) {
        int* helper = index_helper + cols*i;

        //assuming k < cols
        for (int j = 0; j < k; j++) {
            helper[j] = i*cols + j;
            //use heap maybe?
            insert_sort(input, j, helper);
        }
        
        for (int j = k; j < cols; j++) {
            helper[k] = i*cols + j;
            insert_sort(input, k, helper);
            output[helper[k]] = 0.0;
        }

        if (output != input) {//do the copy job
            for (int j = 0; j < k; j++) {
                output[helper[j]] = input[helper[j]];
            }
        }
    }
}

//mask off all elements not in topk by channel
template<typename T>
__global__
void kernel_channelwise_max_filter(const T* input, T* output, const int rows, const int cols, const int channels) {
    CUDA_KERNEL_LOOP(i, rows*channels) {
        const int col_begin = i*cols;
        const int col_end = col_begin + cols;
        int max_j = col_begin;
        T max = input[max_j];
        for (int j = col_begin; j < col_end; j++) {
            if (input[j] > max) {
                max_j = j;
                max = input[j];
            }
            //if output = input
            output[j] = 0.0;
        }

        output[max_j] = max;
    }
}

template<typename T>
void channelwise_topk_filter(cudaStream_t stream, const T* input, T* output, const int k, 
        const int rows, const int cols, const int channels, int* index_helper) {
    if (k >= cols) {
        if (output != input) {
            PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpyAsync(output, input, rows*cols*channels,
                        cudaMemcpyDeviceToDevice, stream));
        }
    } else if (k == 1) {
        kernel_channelwise_max_filter<<<GET_BLOCKS(rows*channels), CUDA_NUM_THREADS, 0, stream>>>(input, output, rows,
                cols, channels);
    } else {
        kernel_channelwise_topk_filter<<<GET_BLOCKS(rows*channels), CUDA_NUM_THREADS, 0, stream>>>(input, output, k,
                rows, cols, channels, index_helper);
    }
}

template<typename T>
//sum all columns in one channel
__global__
void kernel_gpu_sum_bychannels(const int rows, const int cols, const int channels, const T* input,
        T* output) {
    //row*col*channels
    CUDA_KERNEL_LOOP(i, rows*channels) {
        const int begin = i*cols;
        const int end = begin + cols;
        output[i] = 0;
        for (int j = begin; j < end; j++) {
            output[i] += input[j];
        }
    }
}

template<typename T>
void sum_bychannels(cudaStream_t stream, const int rows, const int cols, const int channels, const T* input,
        T* output) {
    kernel_gpu_sum_bychannels<<<GET_BLOCKS(rows*channels), CUDA_NUM_THREADS, 0, stream>>>(rows, cols,
            channels, input, output);
}

template<typename T>
__global__ 
void kernel_stream_kmeans_channelwise(const T* input, const int a, const int b, const int c, const int channels,
        const T* center_sum, const T* center_count, const T power, const T kPrecision, T* output) {
    //a: ins num, b: input dim(per channel), c: center number, channels: input_dim_segments
    CUDA_KERNEL_LOOP(i, a*channels*c) {//b is grouped by channels
        output[i] = 0.0;
        const int begin = i/c*b;
        const int p_id = i%(channels*c);

        for (int j = 0; j < b; j++) {
            T tmp = input[begin + j] - center_sum[j*c*channels + p_id]/center_count[j*c*channels + p_id];
            output[i] += tmp*tmp;
        }

        output[i] = isnan(output[i])?kPrecision:(1.0f/(pow(max(output[i], 0.0), power) + kPrecision));
    }
}

template<typename T>
void stream_kmeans_channelwise(cudaStream_t stream, const T* input, const int a, const int b, const int c, 
        const int channels, const T* center_sum, const T* center_count, const T power, const T kPrecision, T* output) {
    kernel_stream_kmeans_channelwise<<<GET_BLOCKS(a*c*channels), CUDA_NUM_THREADS, 0, stream>>>(input, a, b, c,
            channels, center_sum, center_count, power, kPrecision, output);
}

}  // namespace operators
}  // namespace paddle
