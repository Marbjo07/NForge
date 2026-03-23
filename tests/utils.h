#ifndef UTILS_H
#define UTILS_H

#include "nforge/core/tensor.h"

static Backend backends[] = {Backend::CPU, Backend::CUDA};

static std::string getBackendString(Backend backend) {
    switch (backend) {
        case Backend::CPU:
            return "CPU";
        case Backend::CUDA:
            return "CUDA";
        default:
            return "UNKNOWN";
    }
}

#endif // UTILS_H