#include "tensor.hpp"
#include <iostream>
#include <cassert>

using linalg::Tensor;
using linalg::Shape;

void test() {
    // Test initializer-list constructor
    Tensor<float> v = {1.0f, 2.0f, 3.0f};
    assert(v.shape().size() == 1);
    assert(v.shape()[0] == 3);
    assert(static_cast<float>(v[0]) == 1.0f);
    assert(static_cast<float>(v[1]) == 2.0f);
    assert(static_cast<float>(v[2]) == 3.0f);

    // Test zeros()
    Shape s2{2, 3};
    Tensor<float> z = Tensor<float>::zeros(s2);
    assert(z.shape().size() == 2 && z.shape()[0] == 2 && z.shape()[1] == 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            assert(static_cast<float>(z[i][j]) == 0.0f);
    
    // Test scalar assignment
    z = 5.0f;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            assert(static_cast<float>(z[i][j]) == 5.0f);

    // Test binary operations and broadcasting
    Tensor<float> a = {1.0f, 2.0f, 3.0f};
    Tensor<float> b = {4.0f, 5.0f, 6.0f};
    Tensor<float> c = a + b;
    for (int i = 0; i < 3; ++i)
        assert(static_cast<float>(c[i]) == static_cast<float>(a[i]) + static_cast<float>(b[i]));

    Tensor<float> d = a + 10.0f;
    for (int i = 0; i < 3; ++i)
        assert(static_cast<float>(d[i]) == static_cast<float>(a[i]) + 10.0f);

    Tensor<float> e = 10.0f + a;
    for (int i = 0; i < 3; ++i)
        assert(static_cast<float>(e[i]) == 10.0f + static_cast<float>(a[i]));

    // Test unary operations
    Tensor<float> neg = -a;
    for (int i = 0; i < 3; ++i)
        assert(static_cast<float>(neg[i]) == -static_cast<float>(a[i]));

    // Test exp() and log()
    Tensor<float> zeros3 = Tensor<float>::zeros(Shape{3});
    Tensor<float> ex = zeros3.exp();
    for (int i = 0; i < 3; ++i)
        assert(static_cast<float>(ex[i]) == 1.0f);

    Tensor<float> lg = Tensor<float>::zeros(Shape{3}) + 2.7182818f;
    Tensor<float> lg_res = lg.log();
    for (int i = 0; i < 3; ++i)
        assert(std::abs(static_cast<float>(lg_res[i]) - 1.0f) < 1e-5);

    // Test sum() reduction
    Tensor<float> sum_all = z.sum();
    assert(sum_all.shape().empty());
    assert(static_cast<float>(sum_all) == 2 * 3 * 5.0f);

    // Test reshape, unsqueeze, squeeze
    Tensor<float> r = z.reshape(Shape{3, 2});
    assert(r.shape().size() == 2 && r.shape()[0] == 3 && r.shape()[1] == 2);
    Tensor<float> u = z.unsqueeze(0);
    assert(u.shape().size() == 3 && u.shape()[0] == 1 && u.shape()[1] == 2 && u.shape()[2] == 3);
    Tensor<float> sq = u.squeeze({0});
    assert(sq.shape().size() == 2 && sq.shape()[0] == 2 && sq.shape()[1] == 3);

    // Test matmul
    Tensor<float> m1 = {1.0f, 2.0f, 3.0f, 4.0f};
    m1.assign(m1.reshape(Shape{2,2}));
    Tensor<float> m2 = {5.0f, 6.0f, 7.0f, 8.0f};
    m2.assign(m2.reshape(Shape{2,2}));
    Tensor<float> mm = matmul(m1, m2);
    assert(mm.shape().size() == 2 && mm.shape()[0] == 2 && mm.shape()[1] == 2);
    assert(static_cast<float>(mm[0][0]) == 1*5 + 2*7);
    assert(static_cast<float>(mm[0][1]) == 1*6 + 2*8);
    assert(static_cast<float>(mm[1][0]) == 3*5 + 4*7);
    assert(static_cast<float>(mm[1][1]) == 3*6 + 4*8);

    std::cout << "All tensor tests passed!" << std::endl;
}

int main() {
    {
        backend::BackendGuard guard(backend::BackendType::CpuSingleThread);
        std::cout << "Testing CPU single thread backend" << std::endl;
        test();
    }
    {
        backend::BackendGuard guard(backend::BackendType::CpuMultiThread);
        std::cout << "Testing CPU multi thread backend" << std::endl;
        test();
    }

    return 0;
}