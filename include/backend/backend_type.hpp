#ifndef BACKEND_TYPE_H
#define BACKEND_TYPE_H

namespace backend {

enum class BackendType {
    CpuSingleThread,
    CpuMultiThread,
};

inline std::ostream& operator<<(std::ostream& os, BackendType type) {
    switch (type) {
        case BackendType::CpuSingleThread: return os << "CpuSingleThread";
        case BackendType::CpuMultiThread: return os << "CpuMultiThread";
        default: return os << "Unknown";
    }
}

}

#endif // BACKEND_TYPE_H