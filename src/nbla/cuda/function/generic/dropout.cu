// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// dropout.cu

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/dropout.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_dropout_forward(const int size, const float scale,
                                       const float p, const T *x, T *y,
                                       float *m) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    m[s] = (m[s] > p) ? 1 : 0;
    y[s] = x[s] * m[s] * scale;
  }
}

template <typename T, bool accum>
__global__ void kernel_dropout_backward(const int size, const float scale,
                                        const T *dy, const float *m, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    dx[s] = (accum ? dx[s] : (T)0) + dy[s] * m[s] * scale;
  }
}

template <typename T>
void DropoutCuda<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
  this->mask_.reshape(inputs[0]->shape(), true);
}

template <class T>
void DropoutCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  Variable &mask = this->mask_;
  float *m = mask.cast_data_and_get_pointer<float>(this->ctx_, true);
  curand_generate_rand<float>(curand_generator_, 0.0f, 1.0f, m,
                              inputs[0]->size());
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_dropout_forward, inputs[0]->size(),
                                 this->scale_, this->p_, x, y, m);
}

template <class T>
void DropoutCuda<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Variable &mask = this->mask_;
  const float *m = mask.get_data_pointer<float>(this->ctx_);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_dropout_backward<Tc, true>),
                                   inputs[0]->size(), this->scale_, dy, m, dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_dropout_backward<Tc, false>),
                                   inputs[0]->size(), this->scale_, dy, m, dx);
  }
}
}
