const int MAX_DIMS = 8;

struct TensorLayout {
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    size_t offset;
    size_t rank;
};