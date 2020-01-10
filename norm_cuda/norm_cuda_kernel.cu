#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
template <typename scalar_t>
__global__ void norm_cuda_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weight,
                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> norm,
                          int ou_w, int ou_h,
                          int Wx,
                          int Wy,
                          int Nx,
                          int Ny )
{
  int startX = threadIdx.x * ou_w;
  int startY = threadIdx.y * ou_h;
  if (startX < Wx && startY < Wy) {
    scalar_t sum = 0;
    for (int i = 0; i < ou_w && startX + i < Wx; i++) {
      for (int j = 0; j < ou_h && startY + j < Wy; j++) {
        sum += weight[startX + i][startY + j] * weight[startX + i][startY + j];
      }
    }
    norm[threadIdx.x][threadIdx.y] = sqrt(sum);
  }
}

void norm_cuda(
    torch::Tensor weights,
    torch::Tensor out_norm,
    int ou_w,
    int ou_h)
{
  const auto WeightsSizeX = weights.size(0);
  const auto WeightsSizeY = weights.size(1);
  const auto NormSizeX = out_norm.size(0);
  const auto NormSizeX = out_norm.size(1);
  dim3 threadDim(8, 8);
  dim3 blockDim(((normSizeX - 1) / 8 + 1), ((normSizeY - 1) / 8 + 1));
  
  AT_DISPATCH_FLOATING_TYPES(weights.type(), "norm_cuda", ([&] {
    norm_cuda_kernel<scalar_t><<<blockDim, threadDim>>>(
        weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        out_norm.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        ou_w,
        ou_h,
        WeightsSizeX,
        WeightsSizeY,
        normSizeX,
        normSizeY);
  }));
}