# NForge

A C++ tensor library with a focus on simplicity and ease of use, with optional CUDA support.

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### CUDA support

```bash
cmake .. -DNFORGE_ENABLE_CUDA=on
cmake --build .
```

Requires the CUDA Toolkit. MSBuild is recommended on Windows.

## Quick start

```cpp
#include <iostream>
#include "nforge/tensor.h"

int main() {
    size_t n = 4, m = 5;
    Tensor a({n, m}); // 2D tensor of zeros

    for (size_t i = 0; i < n; i++) {
        a[i] = Tensor({m}, (float)i); // fill each row with its index
    }

    a[0][3] = Tensoor(3.14f);

    a.print();
}
```
### Output:

```text
====================
Tensor[CPU], Data:
0 0 0 3.14 0
1 1 1 1 1
2 2 2 2 2
3 3 3 3 3

Shape: { 4 5 }
====================
```


## API overview

### Construction

| Expression | Description |
|---|---|
| `Tensor({n_1, n_2, ..., n_m}})` | Zero-initialised tensor of shape `{n_1, n_2, ..., n_m}` |
| `Tensor({n, m}, v)` | Tensor filled with value `v` |
| `Tensor(v)` | Scalar tensor wrapping `v` |
| `Tensor(t)` | Copy of tensor `t` |

### Arithmetic

All four operators (`+`, `-`, `*`, `/`) work between:

- Two `Tensor`s, element-wise, with scalar broadcast
- A `Tensor` and a `Tensor::View`
- A `Tensor` (or `View`) and a `float`, in either order

```cpp
Tensor a({4}, 2.0f);

Tensor b = a * 3.0f;       // [6, 6, 6, 6]
Tensor c = 10.0f - a;      // [8, 8, 8, 8]
Tensor d = a + Tensor(1.0f); // scalar broadcast => [3, 3, 3, 3]
```

## Tests

Unit tests are included under `tests/` (Catch2). To run tests after building with CMake:

```bash
ctest --progress
```
