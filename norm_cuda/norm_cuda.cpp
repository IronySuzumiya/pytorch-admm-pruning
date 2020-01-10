#include <torch/extension.h>

// CUDA interface
void norm_cuda(
    torch::Tensor weights,
    torch::Tensor norms,
    int ou_w,
    int ou_h);
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void norm(
    torch::Tensor weights,
    torch::Tensor norms,
    int ou_w,
    int ou_h)
{
  CHECK_INPUT(weights);
  CHECK_INPUT(norms);
  norm_cuda(weights, norms, ou_w, ou_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("norm", &norm, "in-place norm");
}