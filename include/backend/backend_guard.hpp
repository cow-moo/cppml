#ifndef BACKEND_GUARD_H
#define BACKEND_GUARD_H

#include "backend/base.hpp"
#include "config.hpp"

namespace backend {
    inline thread_local BackendType current_backend_type = config::DEFAULT_BACKEND;

    struct BackendGuard {
        BackendType prev;

        BackendGuard(BackendType type) {
            prev = current_backend_type;
            current_backend_type = type;
        }

        ~BackendGuard() {
            current_backend_type = prev;    
        }
    };
}

#endif // BACKEND_GUARD