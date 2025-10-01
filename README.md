# NForge

Tensor is a core data structure in NForge, representing multi-dimensional arrays. As of now, it supports basic arithmetic operations, indexing and assignments.

Indexing returns a view of the original tensor, meaning that modifications to the view will affect the original tensor. This allows assignments of indexed tensors. This allows for furture implementation of strided indexing and slicing.