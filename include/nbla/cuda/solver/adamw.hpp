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

#ifndef __NBLA_CUDA_SOLVER_ADAMW_HPP__
#define __NBLA_CUDA_SOLVER_ADAMW_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/solver/adamw.hpp>

namespace nbla {

template <typename T> class AdamWCuda : public AdamW<T> {
public:
  explicit AdamWCuda(const Context &ctx, float alpha, float beta1, float beta2,
                     float eps, float wd)
      : AdamW<T>(ctx, alpha, beta1, beta2, eps, wd) {}
  virtual ~AdamWCuda() {}
  virtual string name() { return "AdamWCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void update_impl(const string &key, VariablePtr param);
  NBLA_DECL_WEIGHT_DECAY();
  NBLA_DECL_CHECK_INF_GRAD();
  NBLA_DECL_CHECK_NAN_GRAD();
  NBLA_DECL_CHECK_INF_OR_NAN_GRAD();
  NBLA_DECL_SCALE_GRAD();
};
}
#endif
