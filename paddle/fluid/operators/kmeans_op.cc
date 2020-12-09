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

#include "paddle/fluid/operators/kmeans_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class KmeansOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of KmeansOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("KmeansSum"), true,
        platform::errors::InvalidArgument(
            "Input(KmeansSum) of KmeansOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("KmeansCount"), true,
        platform::errors::InvalidArgument(
            "Input(KmeansCount) of KmeansOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of KmeansOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("OutputHelper"), true,
        platform::errors::InvalidArgument(
            "OutputHelper(Out) of KmeansOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ColHelper"), true,
        platform::errors::InvalidArgument(
            "ColHelper(Out) of KmeansOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("TopKHelper"), true,
        platform::errors::InvalidArgument(
            "TopKHelper(Out) of KmeansOp should not be null."));

    auto channels = ctx->Attrs().Get<int>("Channels");

    auto x_dims = ctx->GetInputDim("X");
    auto ins_num = x_dims[0];
    auto sum_dims = ctx->GetInputDim("KmeansSum");
    auto count_dims = ctx->GetInputDim("KmeansCount");
    PADDLE_ENFORCE_EQ(sum_dims, count_dims,
                      platform::errors::InvalidArgument(
                        "Inputs(KmeansSum, KmeansCount)s' shapes differ(%s|%s).", 
                        sum_dims.to_str().c_str(), count_dims.to_str().c_str()));
    
    PADDLE_ENFORCE_EQ(sum_dims[0]*channels, x_dims[1],
                      platform::errors::InvalidArgument(
                        "Input(KmeansSum)'s rows * channels and Input(X)'s cols not same.[%d|%d]", 
                        sum_dims[0]*channels, x_dims[1]));

    ctx->SetOutputDim("Out", {ins_num, sum_dims[1]});
    ctx->SetOutputDim("OutputHelper", {ins_num, sum_dims[1]});
    ctx->SetOutputDim("ColHelper", {ins_num, channels});
    ctx->SetOutputDim("TopKHelper", {ins_num, sum_dims[1]});

    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class KmeansGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Out"), true,
        platform::errors::InvalidArgument("Input(Out) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("KmeansSum"),
                      ctx->GetInputDim("KmeansSum"));
    ctx->SetOutputDim(framework::GradVarName("KmeansCount"),
                      ctx->GetInputDim("KmeansCount"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, "Out"),
                                   ctx.device_context());
  }
};

class KmeansOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of kmeans_op operator.");
    AddInput("KmeansSum",
             "(Tensor) Input tensor of kmeans_op operator.");
    AddInput("KmeansCount",
             "(Tensor) Input tensor of kmeans_op operator.");

    AddOutput("Out", "Output tensor of kmeans_op operator.");
    AddOutput("OutputHelper", "Output tensor of kmeans_op operator.")
        .AsIntermediate();
    AddOutput("ColHelper", "ColHelper tensor of kmeans_op operator.")
        .AsIntermediate();
    AddOutput("TopKHelper", "TopKHelper tensor of kmeans_op operator.")
        .AsIntermediate();

    AddAttr<int>("Channels", "(int, default 1) channels of kmeans_op")
        .SetDefault(1);
    AddAttr<int>("TopK", "(int, default -1) top k of kmeans_op")
        .SetDefault(-1);
    AddAttr<float>("Power", "(float, default 1) power of distance of kmeans_op")
        .SetDefault(1);

    AddComment(R"DOC(
kmeans_op Operator.
This Op can calculate k-nn in ff channelwise, output an indicator normalized vector(scaled by 1/d^power),
and can do kmeans-cluster in bp.
)DOC");
  }
};

template <typename T>
class KmeansGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("kmeans_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("KmeansSum", this->Input("KmeansSum"));
    op->SetInput("KmeansCount", this->Input("KmeansCount"));

    op->SetOutput(framework::GradVarName("KmeansSum"),
                  this->InputGrad("KmeansSum"));
    op->SetOutput(framework::GradVarName("KmeansCount"),
                  this->InputGrad("KmeansCount"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    KmeansGradOpNoNeedBufferVarsInference,  "KmeansSum", "KmeansCount");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(kmeans, ops::KmeansOp,
                  ops::KmeansOpMaker,
                  ops::KmeansGradOpMaker<paddle::framework::OpDesc>,
                  ops::KmeansGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(kmeans_grad, ops::KmeansGradOp,
		ops::KmeansGradOpNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    kmeans,
    ops::KmeansKernel<paddle::platform::CPUDeviceContext, float>,
    ops::KmeansKernel<paddle::platform::CPUDeviceContext, double>);
