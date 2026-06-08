# NForge Introduction

```cpp
#include <nforge/nforge.h>
```

## Construction

```cpp
Tensor a({3, 4});                  // zero-initialized, shape {3, 4}
Tensor b({2, 3}, 1.5f);            // filled with 1.5, shape {2, 3}
Tensor c(3.14f);                   // scalar tensor, shape {1}
Tensor d(b);                       // deep copy
```

## Fill and print

```cpp
Tensor a({2, 3});
a.fillAll(7.0f);                   // all elements = 7
a.fillRand();                      // uniform random in [-1, 1]

a.print();                         // print full tensor
a.print({1});                      // print block starting at (1), of shape {3}
```

## Shape

```cpp
Tensor a({3, 4, 5});
a.getShape();                      // {3, 4, 5}
a.getNumDims();                    // 3
a.getNumElements();                // 60
a.getShape().getDim(1);            // 4
a.getShape().isScalar();           // false (shape {1} is scalar)
```

## Arithmetic

```cpp
Tensor a({3}, 2.0f);
Tensor b({3}, 3.0f);

Tensor r1 = a + b;                 // [5, 5, 5]
Tensor r2 = a - b;                 // [-1, -1, -1]
Tensor r3 = a * b;                 // [6, 6, 6]
Tensor r4 = a / b;                 // [0.67, 0.67, 0.67]

Tensor r5 = a + 1.0f;              // [3, 3, 3], scalar broadcast
Tensor r6 = 10.0f - a;             // [8, 8, 8]

Tensor a2 = a;
a2 += b;                           // in-place, a2 = [5, 5, 5]
```

## Views

```cpp
Tensor a({3, 4}, 0.0f);

Tensor::View row  = a[1];          // shape {4}, row 1
Tensor::View cell = a[1][2];       // shape {1}, element at (1, 2)

auto mat = a[0];                   // shape {4}, first row
```

### Assigning through views

```cpp
Tensor a({3, 4}, 0.0f);

a[0] = Tensor({4}, 7.0f);          // overwrite row 0 with 7s
a[2][3] = Tensor(99.0f);           // set element (2, 3) to 99

Tensor src({4}, 5.0f);
a[1] = src;                        // copy row 1 from another tensor

// a = [[7, 7, 7, 7],
//      [5, 5, 5, 5],
//      [0, 0, 0, 99]]
```

## Strided views

```cpp
Tensor a({6}, 0.0f);
for (size_t i = 0; i < 6; i++) a[i] = Tensor((float)i);

auto evens = a.subsample({2});     // indices 0, 2, 4 -> [0, 2, 4]

// Assigning a strided view mutates the original:
a.subsample({2}) = Tensor({3}, 9.0f);
// a = [9, 0, 9, 0, 9, 0]

// Higher rank stride:
Tensor m({4, 3}, 0.0f);
m.subsample({2, 1}) = Tensor({2, 3}, 1.0f);
// m = [[1, 1, 1],
//      [0, 0, 0],
//      [1, 1, 1],
//      [0, 0, 0]]
```

## Broadcast

Operations broadcast following numpy style. Size 1 dimensions are stretched:

```cpp
Tensor a({3, 1}, 2.0f);            // column vector
Tensor b({1, 4}, 3.0f);            // row vector

Tensor r = a + b;                  // shape {3, 4}, all entries = 5
```

## Reductions

Reductions collapse dimensions `[dim, rank)` into a single value. Result shape is `shape[0:dim]`.

```cpp
Tensor a({3, 4, 5});

auto s = a.sum(1);                 // sum(4*5=20 elements) per leading idx, shape {3}
auto m = a.mean(0);                // mean of all 60 elements, scalar shape {1}
auto v = a.min(2);                 // min per 5-element block, shape {3, 4}
auto x = a.max(1);                 // max per 4x5 block, shape {3}
auto p = a.prod(0);                // product of all elements, scalar
```

## Norm

```cpp
Tensor a({3, 4}, 2.0f);
auto n = a.norm();                 // L2 norm
```

## Matrix multiplication

Supports 2D and 3D tensors.  
2D: `{M, K} @ {K, P} => {M, P}`.  
3D: `{batch, M, K} @ {batch, K, P} => {batch, M, P}`.

```cpp
Tensor A({2, 3}, 1.0f);
Tensor B({3, 4}, 2.0f);
auto C = A.matmul(B);              // shape {2, 4}

// Batched matmul:
Tensor BA({5, 2, 3}, 1.0f);
Tensor BB({5, 3, 4}, 2.0f);
auto BC = BA.matmul(BB);           // shape {5, 2, 4}

// Mixed 2D/3D — 2D is broadcast:
auto C2 = A.matmul(BB);            // A broadcast from {2,3} to {5,2,3}
```

### Functional style

```cpp
auto C = nforge::matmul(A, B)
```

## Comparisons

```cpp
Tensor a({3}, 1.0f);
Tensor b({3}, 2.0f);

bool eq = (a == b);                // false
bool ne = (a != b);                // true

auto lt = a < b;                   // tensor of 0.0/1.0: [1, 1, 1]
auto le = a <= b;                  // [1, 1, 1]
auto gt = a > b;                   // [0, 0, 0]
auto ge = a >= b;                  // [0, 0, 0]

auto close = a.isClose(b, 1e-5f);  // tensor of 0.0/1.0
```

## Backend transfer

```cpp
Tensor a({3, 4}, 1.0f, Backend::CPU);
a.to(Backend::CUDA);               // transfer to CUDA (if enabled)
a.to(Backend::CPU);                // transfer back
```

## Copy to vector

```cpp
Tensor a({2, 2});
a.fillAll(3.0f);
std::vector<float> v = a.toVector(); // {3, 3, 3, 3}, row-major

// For views, resolve to a new contiguous tensor first:
std::vector<float> v = a[1].copy().toVector();
```
