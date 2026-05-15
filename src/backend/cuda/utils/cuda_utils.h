#ifndef NFORGE_CUDA_UTILS_H
#define NFORGE_CUDA_UTILS_H

static constexpr int BLOCK_SIZE = 256;

constexpr int getNumCUDABlocks(size_t count) { return (count + BLOCK_SIZE - 1) / BLOCK_SIZE; }

#endif  // NFORGE_CUDA_UTILS_H