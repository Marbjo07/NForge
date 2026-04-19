#ifndef UTILS_H
#define UTILS_H

#include "nforge/nforge.h"

static constexpr Backend backends[] = {
    Backend::CPU, 

#ifdef NFORGE_WITH_CUDA
    Backend::CUDA
#endif

};

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