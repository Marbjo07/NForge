# Contracts

## Tensors

Always contiguous.

## Views

Must reference a valid tensor or another view.
Does not change reference after construction.
Note: it's possible to deconstruct a tensor with views pointing to it, should be fixed.

## Shape

Reading shape will always return the number of accessable items per dim.
Will always pad such that it has atleast one dimension, scalars have shape {1}.

## Strides

Reading strides will return the number of blocks stepped for each dim, in contrast to numpy which returns the number of elements stepped.

## TensorLayout

Constructor must verify parameters, construction throws if invalid parameters are passed.
Should not be modified after construction.
