#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>
#include "backend/base.hpp"

namespace config {

constexpr size_t MAX_SBO_DIMS = 6;
constexpr backend::BackendType DEFAULT_BACKEND = backend::BackendType::CpuMultiThread;
constexpr size_t DEFAULT_CHUNK_SIZE = 1024;

}

#endif // CONFIG_H