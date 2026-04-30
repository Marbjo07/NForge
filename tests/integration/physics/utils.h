#ifndef UTILS_H
#define UTILS_H

#include "nforge/nforge.h"

static Tensor create2dVector(float x, float y) {
    Tensor a({2}, 0.0f);
    a[0] = x;
    a[1] = y;
    return a;
}  

#endif // UTILS_H