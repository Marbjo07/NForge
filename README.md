# NForge

[![Build](https://github.com/Marbjo07/NForge/actions/workflows/tests.yml/badge.svg)](https://github.com/Marbjo07/NForge/actions)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://marbjo07.github.io/NForge/docs/html/index.html)
[![Benchmarks](https://img.shields.io/badge/benchmarks-live-green)](https://marbjo07.github.io/NForge/dev/bench/)



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
| `NFORGE_BUILD_TESTS` | Toggles build tests | `ON` |


## Quick start

### Code example
```cpp
#include <iostream>
#include "nforge/tensor.h"

int main() {
    size_t n = 4, m = 5;
    Tensor a({n, m}); // 2D tensor of zeros

    for (size_t i = 0; i < n; i++) {
        a[i] = Tensor({m}, (float)i); // fill each row with its index
    }

    a[0][3] = Tensor(3.14f);

    a.print();
}
```
### Output

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

## Documentation

Documentation is built using Doxygen and Github Actions, then hosted on Github Pages on the branch `gh-pages`.  
See [marbjo07.github.io/NForge/docs](https://marbjo07.github.io/NForge/docs/html/index.html)

To generate documentation locally, ensure Doxygen is installed, then from project root:
```bash
doxygen Doxyfile
```

The public API (`Tensor`, `Tensor::View`, `Tensor::Shape`) is fully documented with Doxygen.

See [INTRO.md](INTRO.md) for a quick intro.

## Tests

Unit and integration tests use [Catch2](https://github.com/catchorg/Catch2) and live under `tests/`.

Run after building:

```bash
ctest --progress
```

## Benchmarks

Benchmarks run on merge with `main` branch, after all tests pass.

Current benchmarks are the examples from physics scenarios with default parameters.
The results are published to [marbjo07.github.io/NForge/dev/bench/](https://marbjo07.github.io/NForge/dev/bench/)  


## Code Formatting

NForge uses `clang-format` to enforce consistent formatting.
Formatting is checked automatically on all pull requests and must pass before merging.

### Pre-commit Hook

The easiest way to avoid formatting failures is to install the pre-commit hook,
which checks formatting automatically on every commit:

```bash
# from project root
pip install pre-commit
python -m pre_commit install
```

After installation, any commit that fails formatting will be blocked and fixed in-place.
Stage the changes and commit again to proceed.

Or run clang-format directly through your editor or IDE.
