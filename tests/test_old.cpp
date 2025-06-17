#include <vector>
#include <iostream>
#include <cmath>
#include "autodiff.hpp"
#include "tensor.hpp"
#include "shape.hpp"

using linalg::Tensor;
using linalg::Range;
using linalg::Shape;
using autodiff::Expression;
using autodiff::ComputationGraph;

void test_creation() {
    Tensor<float> t1 = Tensor<float>::zeros({2, 3});
    assert(t1.shape() == std::vector<size_t>({2, 3}));
    
    Tensor<float> t2({{1, 2, 3}, {4, 5, 6}});
    assert(t2.shape() == std::vector<size_t>({2, 3}));
}

void test_indexing() {
    Tensor<float> t({{1, 2, 3}, {4, 5, 6}});
    assert((float)(t[0, 1]) == 2);
    t[1, 2] = 10;
    assert((float)(t[1, 2]) == 10);
}

void test_broadcasting() {
    Tensor<float> t1({1, 2, 3});
    Tensor<float> t2({{1, 2, 3}, {4, 5, 6}});
    Tensor<float> res = t1 + t2;
    assert(res.shape() == std::vector<size_t>({2, 3}));
}

void test_elementwise_ops() {
    Tensor<float> t1({1, 2, 3});
    Tensor<float> t2({4, 5, 6});
    assert(((t1 + t2)[1]) == 7.0f);
    assert(((t1 - t2)[1]) == -3.0f);
    assert(((t1 * t2)[1]) == 10.0f);
    assert(((t1 / t2)[1]) == 2.0f / 5);
}

void test_sum() {
    Tensor<float> t({{1, 2, 3}, {4, 5, 6}});
    Tensor<float> s = t.sum({0});
    assert(s.shape() == std::vector<size_t>({3}));
    assert((float)s[1] == 7);
}

void test_reshape() {
    Tensor<float> t({{1, 2, 3}, {4, 5, 6}});
    Tensor<float> r = t.reshape({3, 2});
    assert(r.shape() == std::vector<size_t>({3, 2}));
}

void test_matmul() {
    Tensor<float> a({{1, 2}, {3, 4}});
    Tensor<float> b({{5, 6}, {7, 8}});
    Tensor<float> c = matmul(a, b);
    assert(c.shape() == std::vector<size_t>({2, 2}));
    assert((float)(c[0, 0]) == 19);
    assert((float)(c[1, 1]) == 50);
}

void test_autodiff_add() {
    ComputationGraph graph;

    Expression<> a({1, 2, 3}, "a", &graph), b({5}, "b", &graph);
    // a.value.print();
    // b.value.print();

    // a.grad->print();
    // b.grad->print();

    // a = a + Tensor<>({1, 2, 3});
    // a.value.print();

    // Expression<> sum = a + b;
    // (*sum.grad) += 1;

    // graph.backward();

    // a.grad->print();
    // b.grad->print();
    // sum.grad->print();
    Expression<> sum = (a + b).sum();
    //sum.value.print();

    sum.backward();
    assert(a.grad()[0] == 1.0f);
    assert(b.grad()[0] == 3.0f);
    //a.grad->print();
    //b.grad->print();
}

void test_autodiff_mul() {
    autodiff::ComputationGraph graph;

    // a: shape [3]
    // b: shape [1] → broadcasted to match a
    autodiff::Expression<> a({1, 2, 3}, "a", &graph);
    autodiff::Expression<> b({5}, "b", &graph);

    // c: elementwise multiplication, shape [3]
    autodiff::Expression<> c = a * b;

    // loss: sum over c, scalar
    autodiff::Expression<> loss = c.sum();

    // run backward pass
    loss.backward();

    // loss.print();
    // a.print();
    // b.print();
    // c.print();

    // Expected gradients:
    // ∂loss/∂a = b = [5, 5, 5]
    // ∂loss/∂b = sum(a) = 6 → because b was broadcast

    assert(a.grad()[0] == 5.0f);
    assert(a.grad()[1] == 5.0f);
    assert(a.grad()[2] == 5.0f);

    assert(b.grad()[0] == 6.0f);
}

void test_autodiff_deep_graph() {
    autodiff::ComputationGraph graph;
    autodiff::Expression<> a({1, 2, 3}, &graph);
    autodiff::Expression<> b({2}, &graph);
    autodiff::Expression<> c({3}, &graph);

    // Deep graph: d = (a + b) * c
    autodiff::Expression<> d = (a + b) * c;

    // loss: sum of d
    autodiff::Expression<> loss = d.sum();

    // run backward pass
    loss.backward();

    // Forward: d = (a + b) * c
    // d = ([1, 2, 3] + 2) * 3 = [3, 4, 5] * 3 = [9, 12, 15]
    // loss = 9 + 12 + 15 = 36

    // Backward:
    // ∂loss/∂d = [1, 1, 1]
    // ∂loss/∂c = sum((a + b)) = sum([3, 4, 5]) = 12
    // ∂loss/∂b = sum(c) = 3 repeated → [3, 3, 3] → reduce = 9
    // ∂loss/∂a = same as b: [3, 3, 3]

    // std::cout << "loss" << std::endl;
    // loss.print();
    // std::cout << "a" << std::endl;
    // a.print();
    // std::cout << "b" << std::endl;
    // b.print();
    // std::cout << "c" << std::endl;
    // c.print();
    // std::cout << "d" << std::endl;
    // d.print();

    assert(a.grad()[0] == 3.0f);
    assert(a.grad()[1] == 3.0f);
    assert(a.grad()[2] == 3.0f);

    assert(b.grad()[0] == 9.0f);
    assert(c.grad()[0] == 12.0f);
}

void test_range_indexing() {
    Tensor<float> t({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    using linalg::Range;

    // Row slice: rows 1 and 2
    auto t1 = t[Range(1, 2)];
    assert(t1.shape() == std::vector<size_t>({2, 3}));
    assert((float)(t1[0, 0]) == 3);
    assert((float)(t1[1, 2]) == 8);

    // Column slice: columns 0 and 1 of all rows
    auto t2 = t[Range(), Range(0, 2)];
    assert(t2.shape() == std::vector<size_t>({3, 2}));
    assert((float)(t2[0, 0]) == 0);
    assert((float)(t2[2, 1]) == 7);

    // Full slice using -1 length (to end)
    auto t3 = t[Range(1, -1), Range(1, -1)];
    assert(t3.shape() == std::vector<size_t>({2, 2}));
    assert((float)(t3[0, 0]) == 4);
    assert((float)(t3[1, 1]) == 8);
}

int main() {
    test_creation();
    test_indexing();
    test_broadcasting();
    test_elementwise_ops();
    test_sum();
    test_reshape();
    test_matmul();
    test_autodiff_add();
    test_autodiff_mul();
    test_autodiff_deep_graph();
    test_range_indexing();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}