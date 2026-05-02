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


| Build Option | Description | Default Value |
|---|---|---|
| `NFORGE_ENABLE_CUDA` | Toggles build CUDA backend | `OFF` |
| `NFORGE_BUILD_BENCHMARKS` | Toggles build benchmarks | `ON` |


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


### Views

A `Tensor::View` is a non-owning window into an existing tensor's data. It shares the same underlying memory, so reads and writes through a view are immediately visible in the original tensor and vice versa.

#### Obtaining a view

`operator[]` steps one dimension into the tensor and returns a view of the remaining dimensions.

```cpp
Tensor a({3, 4}, 0.0f);   // shape {3, 4}

Tensor::View row  = a[1];     // shape {4}, row 1
Tensor::View cell = a[1][2];  // shape {1}, scalar at (1, 2)
```

Chaining `[]` calls navigates deeper into the tensor, one axis at a time:

```cpp
Tensor b({2, 3, 4}, 1.0f);

auto mat  = b[0];       // shape {3, 4}, first matrix
auto row  = b[0][2];    // shape {4},    third row of first matrix
auto elem = b[0][2][3]; // shape {1},     single element
```

#### Assigning through a view

Assigning a `Tensor` (or another `View`) to a view copies data into that window of the original tensor. The original tensor is mutated; the view itself is not reseated.

```cpp
Tensor a({3, 4}, 0.0f);

a[0] = Tensor({4}, 7.0f);   // overwrite row 0 with all 7s

a[2][3] = Tensor(99.0f);    // set element (2, 3) to 99

// Copy row 1 from another tensor
Tensor src({4}, 5.0f);
a[1] = src;

// a = [[7, 7, 7, 7],
//      [5, 5, 5, 5],
//      [0, 0, 0, 99]]
```


Chained assignment works as expected:

```cpp
Tensor a({3}, 1.0f), b({3}, 2.0f), c({3}, 3.0f);
a = b = c;   // both a and b now equal c
```


#### Strided views

`subsample(strides)` returns a view that steps through the tensor with the given per-dimension stride. The view shares the original tensor's memory, so assigning through it writes back directly.

```cpp
Tensor a({6}, 0.0f);
for (size_t i = 0; i < 6; i++) a[i] = Tensor(static_cast<float>(i));
// a = [0, 1, 2, 3, 4, 5]

auto evens = a.subsample({2}); // view of elements at indices 0, 2, 4
```

Because the view is backed by the same memory, assigning to it mutates the original tensor in-place:

```cpp
Tensor a({6}, 0.0f);
// a = [0, 0, 0, 0, 0, 0]

a.subsample({2}) = Tensor({3}, 9.0f);
// a = [9, 0, 9, 0, 9, 0], only the even indices were touched
```

The same principle applies to higher-rank tensors. A stride of `{2, 1}` on a matrix skips every other row while keeping all columns:

```cpp
Tensor m({4, 3}, 0.0f);
m.subsample({2, 1}) = Tensor({2, 3}, 1.0f);
// m = [[1, 1, 1],
//      [0, 0, 0],
//      [1, 1, 1],
//      [0, 0, 0]]
```

## Tests

Unit and integration tests use [Catch2](https://github.com/catchorg/Catch2) and live under `tests/`.

```
tests/
├── unit/                 # tensor construction, arithmetic, views, strides
└── integration/
    └── physics/          # end-to-end physics simulations
```

Run after building:

```bash
ctest --progress
```

## Benchmarks

Benchmarks run on merge with `main` branch, after all tests pass. 

Current benchmarks are the examples from physics scenarios with default parameters.
The results are published to [marbjo07.github.io/NForge/dev/bench/](https://marbjo07.github.io/NForge/dev/bench/)  

## Roadmap

- [ ] Float comparison with configurable epsilon (`approxEqual`) - #14
- [x] Frobenius norm - #15
- [ ] Matrix multiplication - #16
- [ ] Scalar/float–tensor comparison operators - #18
- [x] Reduction operators (sum, mean, max, …) - #19


