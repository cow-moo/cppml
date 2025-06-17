#include <vector>
#include <iostream>
#include <cmath>
#include "autodiff.hpp"
#include "tensor.hpp"
#include "shape.hpp"

using namespace autodiff;
using namespace linalg;

// Helper for floating point comparison
#define ASSERT_CLOSE(a, b, eps) \
    if (std::fabs((a) - (b)) > (eps)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " Assertion failed: " \
                  << #a << " != " << #b << " (" << (a) << " vs " << (b) << ")" \
                  << std::endl; \
        return 1; \
    }

// Test simple addition: dz/dx = 1, dz/dy = 1
int test_add() {
    ComputationGraph graph;
    Tensor<float> t1(Shape{});
    t1 = 2.0f;
    Tensor<float> t2(Shape{});
    t2 = 3.0f;
    Expression<float> x(t1, "x", &graph);
    Expression<float> y(t2, "y", &graph);

    auto z = x + y;
    z.backward();

    float dx = x.grad();
    float dy = y.grad();

    ASSERT_CLOSE(dx, 1.0f, 1e-6f);
    ASSERT_CLOSE(dy, 1.0f, 1e-6f);
    return 0;
}

// Test multiplication: dz/dx = y.value(), dz/dy = x.value()
int test_mul() {
    ComputationGraph graph;
    Tensor<float> t1(Shape{});
    t1 = 4.0f;
    Tensor<float> t2(Shape{});
    t2 = 5.0f;
    Expression<float> x(t1, "x", &graph);
    Expression<float> y(t2, "y", &graph);

    auto z = x * y;
    z.backward();

    float dx = x.grad();
    float dy = y.grad();

    ASSERT_CLOSE(dx, 5.0f, 1e-6f);
    ASSERT_CLOSE(dy, 4.0f, 1e-6f);
    return 0;
}

// Test division: z = x / y
int test_div() {
    ComputationGraph graph;
    Tensor<float> t1(Shape{});
    t1 = 10.0f;
    Tensor<float> t2(Shape{});
    t2 = 2.0f;
    Expression<float> x(t1, "x", &graph);
    Expression<float> y(t2, "y", &graph);

    auto z = x / y;
    z.backward();

    ASSERT_CLOSE(x.grad(), 1.0f / 2.0f, 1e-6f);
    ASSERT_CLOSE(y.grad(), -10.0f / (2.0f * 2.0f), 1e-6f);
    return 0;
}

// Test composite: z = (x * y + y) / x
int test_composite() {
    ComputationGraph graph;
    Tensor<float> t1(Shape{});
    t1 = 3.0f;
    Tensor<float> t2(Shape{});
    t2 = 4.0f;
    Expression<float> x(t1, "x", &graph);
    Expression<float> y(t2, "y", &graph);

    auto z = (x * y + y) / x;
    z.backward();

    // dz/dx = (y*x + y)'_x * 1/x + (x*y + y)*(-1/x^2)
    //       = (y)/x + -(x*y + y)/(x^2)
    float expected_dx = 4.0f/3.0f - (3.0f*4.0f + 4.0f)/(3.0f*3.0f);
    float expected_dy = 1.0f + 1.0f/3.0f;
    ASSERT_CLOSE(x.grad(), expected_dx, 1e-6f);
    ASSERT_CLOSE(y.grad(), expected_dy, 1e-6f);
    return 0;
}

// Test sum reduction and broadcasting
int test_sum() {
    ComputationGraph graph;
    Tensor<float> t({1.0f, 2.0f, 3.0f});
    Expression<float> x(t, "x", &graph);

    auto y = x.sum();
    y.backward();

    for (size_t i = 0; i < 3; ++i) {
        ASSERT_CLOSE(x.grad()[i], 1.0f, 1e-6f);
    }
    return 0;
}

// Test ReLU: gradient 0 for negative, 1 for positive
int test_relu() {
    ComputationGraph graph;
    Tensor<float> t({-1.0f, 0.5f});
    Expression<float> x(t, "x", &graph);

    auto y = x.relu().sum();
    y.backward();

    ASSERT_CLOSE(x.grad()[0], 0.0f, 1e-6f);
    ASSERT_CLOSE(x.grad()[1], 1.0f, 1e-6f);
    return 0;
}

// Test matrix multiplication
int test_matmul() {
    ComputationGraph graph;
    // 2x2 identity and ones
    Tensor<float> I({{1.0f, 0.0f}, {0.0f, 1.0f}});
    Tensor<float> O({{1.0f, 1.0f}, {1.0f, 1.0f}});
    Expression<float> A(I, "A", &graph);
    Expression<float> B(O, "B", &graph);

    auto C = matmul(A, B);
    auto loss = C.sum();
    loss.backward();

    // dC/dA = ones * B^T = row sum of B = [2,2] broadcast
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float a = A.grad()[i, j];
            float b = B.grad()[i, j];
            ASSERT_CLOSE(a, 2.0f, 1e-6f);
            ASSERT_CLOSE(b, 1.0f, 1e-6f);
        }
    }
    return 0;
}

// Test softmax on a 2-vector
int test_softmax() {
    ComputationGraph graph;
    Tensor<float> t({1.0f, 2.0f});
    Expression<float> x(t, "x", &graph);

    auto y = x.softmax().sum();
    y.backward();

    // With softmax, gradient should sum zero across elements
    float g0 = x.grad()[0];
    float g1 = x.grad()[1];
    ASSERT_CLOSE(g0 + g1, 0.0f, 1e-6f);
    return 0;
}

// Test log-softmax
int test_log_softmax() {
    ComputationGraph graph;
    Tensor<float> t({1.0f, 2.0f});
    Expression<float> x(t, "x", &graph);

    auto y = x.log_softmax().sum();
    y.backward();

    // Sum of gradients also zero
    ASSERT_CLOSE(x.grad()[0] + x.grad()[1], 0.0f, 1e-6f);
    return 0;
}

// Test indexing operator
int test_indexing() {
    ComputationGraph graph;
    Tensor<float> t({5.0f, 6.0f, 7.0f});
    Expression<float> x(t, "x", &graph);

    auto y = x[1] * 2.0f;
    y.backward();

    ASSERT_CLOSE(x.grad()[0], 0.0f, 1e-6f);
    ASSERT_CLOSE(x.grad()[1], 2.0f, 1e-6f);
    ASSERT_CLOSE(x.grad()[2], 0.0f, 1e-6f);
    return 0;
}

int main() {
    if (test_add()) return 1;
    if (test_mul()) return 1;
    if (test_div()) return 1;
    if (test_composite()) return 1;
    if (test_sum()) return 1;
    if (test_relu()) return 1;    
    if (test_matmul()) return 1;
    if (test_softmax()) return 1;
    if (test_log_softmax()) return 1;
    if (test_indexing()) return 1;

    std::cout << "All autodiff tests passed!" << std::endl;
    return 0;
}