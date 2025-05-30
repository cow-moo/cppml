#include <iostream>
#include "tensor.hpp"
#include "autodiff.hpp"
#include "module.hpp"

using linalg::Tensor;
using linalg::Range;
using autodiff::Expression;
using autodiff::ComputationGraph;

void test_creation() {
    Tensor<float> t1 = Tensor<float>::zeros({2, 3});
    assert(t1.shape == std::vector<size_t>({2, 3}));
    
    Tensor<float> t2({{1, 2, 3}, {4, 5, 6}});
    assert(t2.shape == std::vector<size_t>({2, 3}));
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
    assert(res.shape == std::vector<size_t>({2, 3}));
}

void test_elementwise_ops() {
    Tensor<float> t1({1, 2, 3});
    Tensor<float> t2({4, 5, 6});
    assert(((t1 + t2)[1]) == 7);
    assert(((t1 - t2)[1]) == -3);
    assert(((t1 * t2)[1]) == 10);
    assert(((t1 / t2)[1]) == 2.0f / 5);
}

void test_sum() {
    Tensor<float> t({{1, 2, 3}, {4, 5, 6}});
    Tensor<float> s = t.sum(0);
    assert(s.shape == std::vector<size_t>({3}));
    assert((float)s[1] == 7);
}

void test_reshape() {
    Tensor<float> t({{1, 2, 3}, {4, 5, 6}});
    Tensor<float> r = t.reshape({3, 2});
    assert(r.shape == std::vector<size_t>({3, 2}));
}

void test_matmul() {
    Tensor<float> a({{1, 2}, {3, 4}});
    Tensor<float> b({{5, 6}, {7, 8}});
    Tensor<float> c = matmul(a, b);
    assert(c.shape == std::vector<size_t>({2, 2}));
    assert((float)(c[0, 0]) == 19);
    assert((float)(c[1, 1]) == 50);
}

void test_autodiff_add() {
    ComputationGraph graph;

    Expression<> a({1, 2, 3}, &graph), b({5}, &graph);
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
    assert((*a.grad)[0] == 1);
    assert((*b.grad)[0] == 3);
    //a.grad->print();
    //b.grad->print();
}

void test_autodiff_mul() {
    autodiff::ComputationGraph graph;

    // a: shape [3]
    // b: shape [1] → broadcasted to match a
    autodiff::Expression<> a({1, 2, 3}, &graph);
    autodiff::Expression<> b({5}, &graph);

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

    assert((*a.grad)[0] == 5);
    assert((*a.grad)[1] == 5);
    assert((*a.grad)[2] == 5);

    assert((*b.grad)[0] == 6);
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

    assert((*a.grad)[0] == 3);
    assert((*a.grad)[1] == 3);
    assert((*a.grad)[2] == 3);

    assert((*b.grad)[0] == 9);
    assert((*c.grad)[0] == 12);
}

void test_range_indexing() {
    Tensor<float> t({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
    using linalg::Range;

    // Row slice: rows 1 and 2
    auto t1 = t[Range(1, 2)];
    assert(t1.shape == std::vector<size_t>({2, 3}));
    assert((float)(t1[0, 0]) == 3);
    assert((float)(t1[1, 2]) == 8);

    // Column slice: columns 0 and 1 of all rows
    auto t2 = t[Range(), Range(0, 2)];
    assert(t2.shape == std::vector<size_t>({3, 2}));
    assert((float)(t2[0, 0]) == 0);
    assert((float)(t2[2, 1]) == 7);

    // Full slice using -1 length (to end)
    auto t3 = t[Range(1, -1), Range(1, -1)];
    assert(t3.shape == std::vector<size_t>({2, 2}));
    assert((float)(t3[0, 0]) == 4);
    assert((float)(t3[1, 1]) == 8);
}

void run_tests() {
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
}

int main() {
    run_tests();

    return 0;
}

// class TwoLayerNet : public Model {
//     TwoLayerNet(Tensor)
// };

// int main() {
//     linalg::Tensor<> a({1, 2, 3});
//     linalg::Tensor<> b({5});
//     //linalg::Tensor<> c = a + b;
//     (a + b).print();
//     // A a;
//     // a.test();
//     // models::Sequential twoLayerNet(3, {layers::Linear(8), layers::ReLU(), layers::Linear(1)}, solver::GradientDescent(0.01), solver::MSE());
//     // twoLayerNet.fit();
// }

/*
TODO
- modules build the computation graph and hold weights/submodules between passes
- implement sum across abritrary axes
- expression class that holds n expression pointers
 - base case is expression that holds a tensor (Parameter)
 - forward pass sets up expression objects, computes final loss fn
 - backprop on loss function traverses the tree generated by expression classes and propagates
 - each expression must cache output as well as gradient of loss wrt output
 - 2 solutions for backprop order
  - compute how many "parent" expressions there are during graph construction. during DFS from loss expression, decrement count whenever expression is reached.
    then, only call backward() when count reaches zero
  - during graph construction, maintain linked list in topo sort order that splices lists from child expressions. then, during backprop iterate through list
 - use concept for expressions
  - value() -> Tensor& (computed and cached on construction)
  - gradient() -> Tensor* (nullptr if doesn't require gradient)
  - backward() (uses cached gradient here and propagates to childrens' gradient caches)
  - other fields
   - value cache (computed and initialized on construction)
   - gradient cache (if gradient is required, initialized to zero)
   - number of parents? or linked list of this expression and child expressions in topo order
 - actually lwk could make this template using this trick for forwards and backwards with pack arguments:
    constexpr auto lambda = [](int a) { return a * 2; }; 
    using U = T<lambda>;
    or T<decltype([](int a) { return a * 2; }){}> obj;

    template<auto forward, auto backward, typename Args...>
    class Expression {};

    template<typename U, typename V>
    AddExpr = Expression<[](Tensor& a, Tensor& b) { return a + b; }, [](Tensor& grad, Tensor* a, Tensor* b) { *a += grad, *b += grad; }, U, V>;
 - store list of learnable Parameters, zero out and use in expressions each time loss is computed. then optimizer does things with Parameter's value and gradient
 - how do we deal with rvalue lvalue? use universal reference T&& and forward() into some processing overload for rvalue and lvalue
  - rvalue should be moved into reference, lvalue should be saved
  - nvm doesnt work, rvalues cannot be moved into reference fields
 - make wrapper class Expression that contains Node pointer
  - allows stuff like x = x + y and means we don't have pointers to stack variables
- consider doing template <auto op> for passing functions. does not allow capture however
- simd: during axes recursion stop when subtensors are full (stored contiguously) and do simd operations
 - scalar operations should have built in operations to broadcast. research further
 - when broadcasting, do profiling for relative size between tensors. can try striding and adding or adding smaller tensor many times
 - think about when both tensors are being duplicated
 - could also stop earlier when theres a fixed stride (iff tail axes are fixed?)
 - use neon
- try to join consecutive operations? a + (b * 2) gets evaluated once
- random init
- implement tail recursion elimination
- research how to optimize/parallelize these elementwise operations
- .sum()
- figure out scalar operations
- matmul
- reshape
 - might need to do copy operations when slices occur
 - in place or return?
- numpy also does copies when doing arbitrary slices, seems to also keep views with Range objects
- figure out elementwise assignment (make TensorSlice class with elementwise assignment and Tensor conversion operator for implicit casting?)
- idea: make the TensorSlice class just modify indexing in place? (doesn't really work)
- other idea: have elementwise operation just keep track of running index offsets so it doesn't need to make new Tensor slices
 - probably more correct
- main problem is that we're copying axisIndices/shape which mostly aren't changing and shared ptr data overhead?
*/