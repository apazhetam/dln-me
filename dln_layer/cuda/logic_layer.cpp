// Code modified from the "difflogic - A Library for Differentiable Logic Gate Networks" GitHub folder:
// https://github.com/Felix-Petersen/difflogic/blob/main/difflogic/cuda/difflogic.cpp
// Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver.
// Deep Differentiable Logic Gate Networks.
// Conference on Neural Information Processing Systems (NeurIPS).
// 2022.

#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>



namespace py = pybind11;

torch::Tensor logic_layer_cuda_forward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);

torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
);

std::tuple<torch::Tensor, torch::Tensor> logic_layer_cuda_backward_ab(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y
);

torch::Tensor logic_layer_cuda_eval(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_forward(a, b, w);
        },
        "logic layer forward (CUDA)");
    m.def(
        "backward_w", [](torch::Tensor a, torch::Tensor b, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_w(a, b, grad_y);
        },
        "logic layer backward w (CUDA)");
    m.def(
        "backward_ab",
        [](torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_ab(a, b, w, grad_y);
        },
        "logic layer backward a and b (CUDA)");
    m.def(
        "eval",
        [](torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_eval(a, b, w);
        },
        "logic layer eval (CUDA)");
}
