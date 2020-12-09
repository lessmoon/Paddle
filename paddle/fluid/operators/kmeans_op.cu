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

#include <cublas.h>
#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/kmeans_op.cu.h"
#include "paddle/fluid/operators/kmeans_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename DeviceContext, typename T>
class KmeansCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");
    auto *kmeans_sum = ctx.Input<Tensor>("KmeansSum");
    auto *kmeans_count = ctx.Input<Tensor>("KmeansCount");

    int channels = ctx.Attr<int>("Channels");
    int top_k = ctx.Attr<int>("TopK");
    T power = T(ctx.Attr<float>("Power")/2);

    auto *Out = ctx.Output<Tensor>("Out");

    auto& dev_ctx = ctx.cuda_device_context();

    // check dims
    auto x_dims = X->dims();
    auto ins_num = x_dims[0];
    auto para_dims = kmeans_sum->dims();
    
    PADDLE_ENFORCE_EQ(
        kmeans_sum->dims(), kmeans_count->dims(),
        platform::errors::InvalidArgument("Input(KmeansCount) and Input(KmeansSum) shapes[%s|%s] not same",
            kmeans_sum->dims().to_str().c_str(), kmeans_count->dims().to_str().c_str()));

    PADDLE_ENFORCE_EQ(
        para_dims[0]*channels, x_dims[1],
        platform::errors::InvalidArgument("Input(KmeansSum)'s rows * channels and Input(X)'s cols not same.[%d|%d]",
            para_dims[0]*channels, x_dims[1]));
 
    auto& output_helper = *ctx.Output<Tensor>("OutputHelper");
    auto& col_helper = *ctx.Output<Tensor>("ColHelper");

    const T *in_data = X->data<T>();
    const T *sum_data = kmeans_sum->data<T>();
    const T *count_data = kmeans_count->data<T>();
    
    output_helper.mutable_data<T>(ctx.GetPlace());
    Out->mutable_data<T>(ctx.GetPlace());
    T *output_helper_data = output_helper.data<T>();
    T *out_data = Out->data<T>();

    stream_kmeans_channelwise(ctx.cuda_device_context().stream(),
            in_data,
            int(ins_num), int(X->dims()[1]/channels), int(kmeans_sum->dims()[1]/channels), int(channels),
            sum_data, count_data, power, T(1e-7), output_helper_data);
 
    if (top_k > 0) {
        auto& top_k_helper = *ctx.Output<Tensor>("TopKHelper");
        top_k_helper.mutable_data<int>(ctx.GetPlace());

        channelwise_topk_filter(ctx.cuda_device_context().stream(),
                output_helper_data, output_helper_data,
                top_k,
                output_helper.dims()[0], output_helper.dims()[1]/channels, channels,
                top_k_helper.data<int>());
    }
    
    col_helper.mutable_data<T>(ctx.GetPlace());
    sum_bychannels(ctx.cuda_device_context().stream(),
            output_helper.dims()[0], output_helper.dims()[1]/channels, channels,
            output_helper_data, col_helper.data<T>());

    colwise_div(ctx.cuda_device_context().stream(),
            output_helper.dims()[0], output_helper.dims()[1]/channels, channels,
            output_helper_data, col_helper.data<T>(), out_data);

  }
};

template <typename DeviceContext, typename T>
class KmeansGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");
    auto *Out = ctx.Input<Tensor>("Out");
    auto& dev_ctx = ctx.cuda_device_context();
    int channels = ctx.Attr<int>("Channels");

    auto *kmeans_sum_grad = ctx.Output<Tensor>(framework::GradVarName("KmeansSum"));
    auto *kmeans_count_grad = ctx.Output<Tensor>(framework::GradVarName("KmeansCount"));
    
    Tensor row_helper;
    row_helper = ctx.AllocateTmpTensor<T, DeviceContext>(
            {1, kmeans_sum_grad->dims()[1]}, dev_ctx);
 
    row_helper.mutable_data<T>(ctx.GetPlace());
    kmeans_sum_grad->mutable_data<T>(ctx.GetPlace());
    kmeans_count_grad->mutable_data<T>(ctx.GetPlace());
    
    math::SetConstant<DeviceContext, T>()(dev_ctx, kmeans_count_grad, static_cast<T>(0));
    math::ColwiseSum<DeviceContext, T> col_sum;
    col_sum(dev_ctx, *Out, &row_helper);
    math::RowwiseAdd<DeviceContext, T> add;
    add(dev_ctx, *kmeans_count_grad, row_helper, kmeans_count_grad);
   
    T alpha = 1;
    T beta = 0;

    cublasOperation_t cuTransA = CUBLAS_OP_T;
    cublasOperation_t cuTransB = CUBLAS_OP_N;

    //Rowwise
    const int strideA = X->dims()[1]/channels;
    const int strideB = Out->dims()[1]/channels;
    const int strideC = kmeans_sum_grad->dims()[1]/channels;
    const int lda = X->dims()[1];
    const int ldb = Out->dims()[1];
    const int ldc = kmeans_sum_grad->dims()[1];
    dev_ctx.CublasCall([&](cublasHandle_t handle) {
        math::CUBlas<T>::GEMM_STRIDED_BATCH(handle, cuTransB, cuTransA, Out->dims()[1]/channels,
            X->dims()[1]/channels, X->dims()[0], &alpha,
            Out->data<T>(), ldb, strideB, X->data<T>(), lda, strideA, &beta, kmeans_sum_grad->data<T>(),
            ldc, strideC, channels);
    });

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(kmeans,
                        ops::KmeansCUDAKernel<GPUCtx, float>,
                        ops::KmeansCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(kmeans_grad,
                        ops::KmeansGradOpCUDAKernel<GPUCtx, float>,
                        ops::KmeansGradOpCUDAKernel<GPUCtx, double>);
