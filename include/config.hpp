#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>
#include "backend/base.hpp"

constexpr size_t MAX_SBO_DIMS = 6;
constexpr backend::BackendType DEFAULT_BACKEND = backend::BackendType::CpuSingleThread;

#endif // CONFIG_H