# README for numpy_extensions Class

## Overview

The numpy_extensions class provides a collection of class methods that extend the functionality of numpy arrays. These methods include array manipulation techniques such as checking for the presence of elements, replacing elements, multi-indexing, flattening multi-dimensional arrays, one-hot encoding, and performing spectral clustering. Below are descriptions and examples for each method provided in the class.

## Methods

### `contains(a: np.ndarray, b: np.ndarray) -> np.ndarray`

Checks if elements of array `b` are present in array `a`.
This implementation is espacialy usefull for working with multi dimensional arrays.

#### Example

```python
import numpy as np
from numpy_extensions import numpy_extensions as npe

a = np.array([0,1,2])
b = np.array([0,3,1,4])

print(npe.contains(a, b))
# Output: [True False True False]
```
```python
import numpy as np
from numpy_extensions import numpy_extensions as npe

a = np.array([[0,1],[1,2]])
b = np.array([[0,1],[0,2],[1,2]])

print(npe.contains(a, b))
# Output: [True False True]
```
```python
import numpy as np
from numpy_extensions import numpy_extensions as npe

a = np.array([[[0,1],[2,3]], [[4,5],[6,7]]])
b = np.array([[[0,1],[2,3]], [[4,5],[6,0]]])

print(npe.contains(a, b))
# Output: [True False]
```

### `replace(a: np.ndarray, values: any, replacements: any) -> np.ndarray`

Replaces specified values in array `a` with the corresponding replacements.

#### Examples

```python
# Single value replacement
print(npe.replace(np.array([1, 2, 3, 2, 1]), 1, 0))
# Output: [0 2 3 2 0]

# Multiple values replacement
print(npe.replace(np.array([1, 2, 3, 2, 1]), [1, 2], [-1, -2]))
# Output: [-1 -2 3 -2 -1]

# Multi-dimensional array replacement
a = np.array([[1, 2], [1, 3], [2, 4]])
print(npe.replace(a, [1, 2, 3, 4], [0, 1, 2, 3]))
# Output: [[0 1] [0 2] [1 3]]
```

<!-- ### `setitem_multi_index(a: np.ndarray, ids: np.ndarray, values: np.ndarray) -> np.ndarray`

Sets elements in `a` at multi-dimensional indices `ids` to `values`.

#### Example

```python
a = np.zeros((4, 4))
ids = np.array([[0, 1], [2, 3]])
values = np.array([10, 20])
print(npe.setitem_multi_index(a, ids, values))
```

### `getitem_multi_index(a: np.ndarray, ids: np.ndarray) -> np.ndarray`

Retrieves elements from `a` at multi-dimensional indices `ids`.

#### Example

```python
a = np.arange(16).reshape((4, 4))
ids = np.array([[0, 1], [2, 3]])
print(npe.getitem_multi_index(a, ids))
``` -->

<!-- ### `get_basis(dims: tuple[int, int, int]) -> np.ndarray`

Calculates the basis for flattening a multi-dimensional array given its dimensions.

#### Example

```python
dims = (3, 3, 3)
print(npe.get_basis(dims))
``` -->

### `flatten_multi_index(ids: np.ndarray, dims, dtype: np.dtype=None, axis: int=1) -> np.ndarray`

Encodes multiple vectors of n dimensions and type int to a scalar based on the provided dimensions.

#### Examples

```python
# Flattening a 2D index array
dims = (2, 2)
vectors = np.array([[0,0],[0,1],[1,0],[1,1]])
print(npe.flatten_multi_index(vectors, dims))
# Output: [0. 1. 2. 3.]

# Flattening a 3D index array
dims = (4, 4, 4)
vec = np.array([[0, 1, 2], [1, 2, 3]])
print(npe.flatten_multi_index(vec, dims))
# Output: [6. 27.]
```

### `one_hot(values: np.ndarray, class_count: int=None) -> np.ndarray`

Performs one-hot encoding on the input array `values`.

#### Example

```python
values = np.array([0, 1, 2, 1])
print(npe.one_hot(values))
# Output: [[1., 0., 0.],
#          [0., 1., 0.],
#          [0., 0., 1.],
#          [0., 1., 0.]]
```

### `flatten(a: np.ndarray, axis: int=0) -> np.ndarray`

Flattens an array along the specified axis.

#### Example

```python
a = np.array([[1, 2], [3, 4]])
print(npe.flatten(a))
# Output: [1 2 3 4]
```

### `getitem_nd(a: np.ndarray, id_mtx: np.ndarray) -> np.ndarray`

Generates a view from a multidimensional id-array, substituting every id in `id_mtx` with its related item in `a`.

#### Examples

```python
# Simple case
a = np.array([1, 2, 3])
id_mtx = np.array([2, 0, 1])
print(npe.getitem_nd(a, id_mtx))
# Output: [3 1 2]

# Multi-dimensional case
id_mtx = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
print(npe.getitem_nd(a, id_mtx))
# Output: [[1 2 3] [2 3 1] [3 1 2]]
```

<!-- ### `setitem_nd(a: np.ndarray, id_mtx: np.ndarray, values: np.ndarray) -> np.ndarray`

Sets elements in `a` specified by multi-dimensional indices `id_mtx` to `values`.

#### Example

```python
a = np.zeros((3, 3))
id_mtx = np.array([[0, 1], [1, 2]])
values = np.array([10, 20])
print(npe.setitem_nd(a, id_mtx, values))
``` -->

<!-- ### `spectral_clustering(laplacian_matrix: np.ndarray, k_clusters: int, k_eigenvectors: int=None) -> np.ndarray`

Performs spectral clustering on the Laplacian matrix.

#### Example

```python
laplacian_matrix = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
k_clusters = 2
print(npe.spectral_clustering(laplacian_matrix, k_clusters))
``` -->

### `relabel(a: np.ndarray) -> np.ndarray`

Relabels elements in `a` to integers, ensuring identical values receive the same integer, starting from 0.

#### Example

```python
a = np.array([3, 1, 2, 1, 3, 2, 4])
print(npe.relabel(a))
# Output: [0 1 2 1 0 2 3]
```


## unittest
run tests in console
```
python -m unittest
```