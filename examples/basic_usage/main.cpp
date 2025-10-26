// Basic usage example for TensorVar and tensor<> helper

#include "include/ml/common/TensorVar.h"
#include <iostream>

using namespace hahaha;
using core::ds::Vector;

int main()
{
    // 1) Create a scalar tensor using the helper factory
    auto s = tensor<float>(3.14f, "pi");
    std::cout << "Scalar size: " << s->size() << ", value: " << s->data()[0]
              << std::endl;

    // 2) Create a 1D tensor with shape {3}
    auto v = tensor<float>({3});
    std::cout << "Vector shape: " << v->shape().toString()
              << ", size: " << v->size() << std::endl;

    // Fill values manually
    v->set({0}, 1.0f);
    v->set({1}, 2.0f);
    v->set({2}, 3.0f);
    std::cout << "Vector sum: " << v->sumValue() << std::endl;

    // 3) Create a 2x2 tensor with data (shape and data are two arguments)
    auto m = tensor<float>({2, 2},
                           {1.0f, 2.0f,
                                 3.0f, 4.0f},
                           "m");
    std::cout << "Matrix shape: " << m->shape().toString()
              << ", sum: " << m->sumValue() << std::endl;

    // 4) Use pointer-like forwarding (transpose)
    auto mt = m->transpose();
    std::cout << "Transposed shape: " << mt.shape().toString() << std::endl;

    return 0;
}


