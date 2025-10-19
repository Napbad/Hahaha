// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad

#define CHECK(cuda_call) \
do { \
cudaError_t err = cuda_call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error: %s at %s:%d\n", \
cudaGetErrorString(err), __FILE__, __LINE__); \
exit(EXIT_FAILURE); \
} \
} while (0)
#include "defines/h3defs.h"

__global__ void vectorAddKernel(const float* d_a, const float* d_b, float* d_c, int n) {
    if (const hahaha::sizeT idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}
