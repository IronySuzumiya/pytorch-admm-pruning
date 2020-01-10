from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sjq_utils',
    ext_modules=[
        CUDAExtension('norm_cuda', [
            'norm_cuda.cpp',
            'norm_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })