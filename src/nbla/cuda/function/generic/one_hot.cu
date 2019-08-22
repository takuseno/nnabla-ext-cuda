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

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/one_hot.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename TI, typename T>
__global__ void kernel_one_hot(const int num, const int size, const TI *x, T *y,
                               const int *shape, const int dim) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    int addr = 0;
    Size_t s = 1;
    for (int i = dim - 1; i >= 0; --i) {
      addr += x[idx * dim + i] * s;
      s *= shape[i];
    }
    y[idx * size + addr] = (T)1;
  }
}

template <typename TI, typename T>
void OneHotCuda<TI, T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);
  OneHot<TI, T>::setup_impl(inputs, outputs);
}

template <typename TI, typename T>
void OneHotCuda<TI, T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);
  const TIcu *x = inputs[0]->get_data_pointer<TIcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  int *shape_cpu = new int[this->shape_.size()];
  for (int i = 0; i < this->shape_.size(); ++i) {
    shape_cpu[i] = this->shape_[i];
  }

  CudaCachedArray cushape(sizeof(int) * this->shape_.size(), dtypes::BYTE,
                          this->ctx_);
  void *cushape_ptr = cushape.pointer<void>();
  NBLA_CUDA_CHECK(cudaMemcpy((int *)cushape_ptr, shape_cpu,
                             sizeof(int) * this->shape_.size(),
                             cudaMemcpyHostToDevice));
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_one_hot, this->num_, this->size_, x, y,
                                 (int *)cushape_ptr, this->dim_);

  delete[] shape_cpu;
}
}
