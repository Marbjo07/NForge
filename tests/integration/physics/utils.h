#ifndef UTILS_H
#define UTILS_H

#include "nforge/nforge.h"

static Tensor create2dVector(float x, float y) {
    Tensor a({2}, 0.0f);
    a[0] = x;
    a[1] = y;
    return a;
}  

static bool isSimilar(Tensor a, Tensor b) {
    Tensor err = a - b;
    err *= err;

    if (err.mean().toVector()[0] < 0.0001) return true;
    else return false;
}

#endif // UTILS_H