// Copyright (c) 2019 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Include nnabla header files

#include <nbla/context.hpp>
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
using namespace std;
using namespace nbla;

#include <siamese_training.hpp>

/******************************************/
// Example of mnist training
/******************************************/
int main(int argc, char *argv[]) {

  // Create a context (the following setting is recommended.)
  nbla::init_cudnn();
  nbla::Context ctx{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};

  // Execute training
  if (!siamese_training(ctx)) {
    return (-1);
  }

  return 0;
}
