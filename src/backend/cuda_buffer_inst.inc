template class CudaBuffer<float>;
template class CudaBuffer<unsigned long>;
template class CudaBuffer<bool>;
template class CudaBuffer<int>;
template class CudaBuffer<uint8_t>;

#define INSTANTIATE_APPLY_BINARY(T, U, V) \
template void CudaBuffer<T>::apply_binary<U, V>( \
    const Shape&, const Strides&, size_t, \
    DeviceBuffer<U>*, const Strides&, size_t, \
    DeviceBuffer<V>*, const Strides&, size_t, \
    BinOp); \
template void CudaBuffer<T>::apply_binary<U, V>( \
    const Shape&, const Strides&, size_t, \
    DeviceBuffer<U>*, const Strides&, size_t, \
    V, BinOp);

INSTANTIATE_APPLY_BINARY(float, float, float)
INSTANTIATE_APPLY_BINARY(unsigned long, unsigned long, unsigned long)
INSTANTIATE_APPLY_BINARY(int, int, int)
INSTANTIATE_APPLY_BINARY(uint8_t, uint8_t, uint8_t)

INSTANTIATE_APPLY_BINARY(bool, float, float)
INSTANTIATE_APPLY_BINARY(bool, unsigned long, unsigned long)
INSTANTIATE_APPLY_BINARY(bool, int, int)

#define INSTANTIATE_APPLY_UNARY(T, U) \
template void CudaBuffer<T>::apply_unary<U>( \
    const Shape&, const Strides&, size_t, \
    DeviceBuffer<U>*, const Strides&, size_t, \
    UnOp);

INSTANTIATE_APPLY_UNARY(float, float)
INSTANTIATE_APPLY_UNARY(unsigned long, unsigned long)
INSTANTIATE_APPLY_UNARY(bool, bool)
INSTANTIATE_APPLY_UNARY(int, int)
INSTANTIATE_APPLY_UNARY(uint8_t, uint8_t)

INSTANTIATE_APPLY_UNARY(float, bool)
INSTANTIATE_APPLY_UNARY(unsigned long, bool)
INSTANTIATE_APPLY_UNARY(int, bool)

INSTANTIATE_APPLY_UNARY(float, unsigned long)
INSTANTIATE_APPLY_UNARY(float, int)
INSTANTIATE_APPLY_UNARY(float, uint8_t)

#define INSTANTIATE_ARG_REDUCE(U) \
template void CudaBuffer<size_t>::arg_reduce<U>( \
    const Shape&, const Strides&, size_t, \
    const DeviceBuffer<U>*, const Strides&, size_t, \
    size_t, ArgRedOp);

INSTANTIATE_ARG_REDUCE(float)