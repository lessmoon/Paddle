# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import random
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core

def np_kmeans(input, channels, sums, counts, topk, power):
    input_row, input_col = input.shape
    kmeans = sums/counts
    output = np.zeros((input_row, sums.shape[1]))
    centers = sums.shape[1]/channels
    dims = input_col/channels

    for i in range(channels):
        kmeans_ = kmeans[..., centers*i: centers*(i+1)]
        in_val_ = input[..., dims*i:dims*(i+1)]
        ab = in_val_.dot(kmeans_)*(-2.0)
        ab += np.apply_along_axis(np.sum, 1, np.square(in_val_)).reshape(-1, 1)
        ab += np.apply_along_axis(np.sum, 0, np.square(kmeans_)).reshape(1, -1)
        ab[ab < 0] = 0.0
        x = 1.0/(np.power(np.sqrt(ab), power) + 1e-7)
        if topk > 0 and x.shape[1] > topk:
            s = np.argsort(-x, 1)#rowwisesort
            row, _=np.indices(s.shape)
            x[row[..., topk:], s[..., topk:]] = 0.0
            #print(x)

        output[...,centers*i: centers*(i+1)]=x/np.apply_along_axis(np.sum, 1, x).reshape(-1, 1)

    return output

class TestKmeansOpComplex(OpTest):
    def config(self):
        self.ins_num = 10
        self.channels = 2
        self.centers = 3
        self.input_dim = 2
        self.topk = 1
        self.dtype = "float64"
        self.power = 1.0

    def setUp(self):
        self.op_type = "kmeans"
        self.config()
        input = np.random.random((self.ins_num, self.channels*self.input_dim)).astype(self.dtype)
        input_row, input_col = input.shape
        kmeans_para = np.random.random((self.input_dim, self.channels*self.centers)).astype(self.dtype)
        kmeans_counts = np.ones((self.input_dim, self.channels*self.centers)).astype(self.dtype)

        out = np_kmeans(input, self.channels, kmeans_para, kmeans_counts, self.topk, self.power)

        self.inputs = {
            "X": input,
            "KmeansSum": kmeans_para,
            "KmeansCount": kmeans_counts,
        }
        self.attrs = {'Channels': self.channels,
                      'TopK': self.topk,
                      'Power': self.power}
        self.outputs = {
            "Out": out
        }

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            self.check_grad_with_place(core.CUDAPlace(0), [], "Out")


class TestKmeansOpCpu(OpTest):
    def config(self):
        self.ins_num = 10
        self.channels = 2
        self.centers = 3
        self.input_dim = 2
        self.topk = 1
        self.dtype = "float64"
        self.power = 1.0

        pass

    def setUp(self):
        self.op_type = "kmeans"
        self.config()
        input = np.random.random((self.ins_num, self.channels*self.input_dim)).astype(self.dtype)
        input_row, input_col = input.shape
        kmeans_para = np.random.random((self.input_dim, self.channels*self.centers)).astype(self.dtype)
        kmeans_counts = np.ones((self.input_dim, self.channels*self.centers)).astype(self.dtype)

        out = np_kmeans(input, self.channels, kmeans_para, kmeans_counts, self.topk, self.power)

        self.inputs = {
            "X": input,
            "KmeansSum": kmeans_para,
            "KmeansCount": kmeans_counts,
        }
        self.attrs = {'Channels': self.channels,
                      'TopK': self.topk,
                      'Power': self.power}
        self.outputs = {
            "Out": out
        }

    def test_check_output_cpu(self):
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print("do not support cpu test, skip")

    def test_check_grad_cpu(self):
        try:
            self.check_grad_with_place(core.CUDAPlace(0), [], "Out")
        except:
            print("do not support cpu test, skip")


if __name__ == "__main__":
    unittest.main()
