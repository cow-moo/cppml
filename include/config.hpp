#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>
#include "backend/backend_type.hpp"

namespace config {

constexpr size_t MAX_DIMS = 6;
constexpr backend::BackendType DEFAULT_BACKEND = backend::BackendType::CpuSingleThread;

}

#endif // CONFIG_H