# NForge

Tensor library for C++ with a focus on simplicity and ease of use. Features include:

- Creation from shapes (scalars, vectors, matrices and higher-rank tensors)
- Element-wise arithmetic (+, -, *, /)
- Indexing that returns views (e.g. `a[0]` returns a view of the first row of `a`) and supports assignment
- Views that can be assigned to (e.g. `a[0] = Tensor({m}, 1.0f)` sets the first row of `a` to all ones)
- Backend abstraction allows for easy switching between CPU and CUDA implementations.

## Build

From the `NForge` root directory:

```bash
mkdir build
cd build
cmake ..
```

## Example

A small example, building a 2D tensor and indexing into it.

```cpp
#include <iostream>
#include "nforge/tensor.h"

int main() {
    size_t n = 4, m = 5;
    Tensor a({n, m}); // 2D tensor of shape (n, m) filled with zeros

    for (int i = 0; i < n; i++) {
        a[i] = Tensor({m}, static_cast<float>(i)); // Each row is filled with the row index
    }

    // Set a single value
    a[0][3] = Tensor(3.14f);

    a.print();

    return 0;
}
```

The output:

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

## Tests

Unit tests are included under `tests/` (Catch2). To run tests after building with CMake:

```bash
ctest --progress
```
