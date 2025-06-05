    // struct Iterator {
    //     const U* data;
    //     Shape shape;
    //     std::array<size_t, MAX_SBO_DIMS> strides;
    //     std::array<size_t, MAX_SBO_DIMS> idxs;
    //     size_t flatIdx;

    //     Iterator(const Tensor& t) : data(t.data_.get()) {
    //         shape = t.shape_;
    //         for (size_t i = 0; i < t.strides_.size(); i++)
    //             strides[i] = t.strides_[i];
    //         flatIdx = t.offset_;
    //     }

    //     U& operator*() {
    //         return data[flatIdx];
    //     }

    //     Iterator& operator++() {
    //         for (int i = shape.size() - 1; i >= 0; i--) {
    //             flatIdx += strides[i];
    //             if (++idxs[i] == shape[i]) {
    //                 idxs[i] = 0;
    //                 flatIdx -= strides[i] * shape[i];
    //             }
    //             else break;
    //         }
    //         return *this;
    //     }

    //     Iterator& operator+=(size_t n) {
    //         idxs[shape.size() - 1] += n;
    //         flatIdx += n * strides[shape.size() - 1];

    //         for (int i = shape.size() - 1; i >= 1; i--) {
    //             if (idxs[i] >= shape[i]) {
    //                 size_t num = idxs[i] / shape[i];
    //                 idxs[i] -= num * shape[i];
    //                 flatIdx -= num * shape[i] * strides[i];
    //                 idxs[i - 1] += num;
    //                 flatIdx += num * strides[i - 1];
    //             }
    //             else break;
    //         }

    //         return *this;
    //     }
        
    //     Iterator operator+(size_t n) {
    //         Iterator res(*this);
    //         res += n;
    //         return res;
    //     }

    //     bool operator==(const Iterator& other) const {
    //         return other.data == data && other.flatIdx == flatIdx;
    //     }
    // };