from __future__ import annotations
import numpy as np
from collections.abc import Iterable
from sklearn.cluster import KMeans

class numpy_extensions():
    @classmethod
    def contains(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(np.shape(a)) == 1 and len(np.shape(a)) == 1:
            return np.in1d(b, a)

        dims = (np.maximum(np.max(a),np.max(b))+1,)*(np.sum(np.shape(a)[1:]))
        hash_a = cls.flatten_multi_index(a, dims)
        hash_b = cls.flatten_multi_index(b, dims)
        return np.in1d(hash_b, hash_a)


    @classmethod
    def replace(cls, a: np.ndarray, values: any, replacements: any) -> np.ndarray:
        '''
        # Notes

        Replaces given values in a by given replacements.

        # Examples

        >>> ne.replace([1,2,3,2,1], 1, 0)
        array([0, 2, 3, 2, 0])

        multiple values
        >>> ne.replace([1,2,3,2,1], [1,2], [-1,-2])
        array([-1, -2,  3, -2, -1])

        multiple dimensions
        >>> a = np.array([[1,2], [1,3], [2,4]])
        >>> a
        array([[1, 2],
                [1, 3],
                [2, 4]])
        >>> ne.replace(a, [1,2,3,4], [0,1,2,3])
        array([[0, 1],
                [0, 2],
                [1, 3]])
        '''
        if len(a) == 0:
            return np.copy(a)
        if not isinstance(values, Iterable) and not isinstance(replacements, Iterable):
            result = np.copy(a)
            result[np.equal(a, values)] = replacements
            return result

        if isinstance(values, Iterable) and not isinstance(replacements, Iterable):
            replacements = np.full(values.shape, replacements)
        if type(replacements) is not np.ndarray:
            replacements = np.array(replacements)

        result = np.copy(a)
        shape = result.shape
        result = result.flatten()

        d = dict(zip(values, replacements))
        def r(x): return d[x]

        mask = np.in1d(result, values)
        result[mask] = np.frompyfunc(r, 1, 1)(result[mask])
        result = result.reshape(shape)
        return result.astype(replacements.dtype)

    @classmethod
    def setitem_multi_index(cls, a: np.ndarray, ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        flat_a = a.flatten()
        flat_ids = np.ravel_multi_index(np.transpose(ids), a.shape)

        flat_a[flat_ids] = values

        return flat_a.reshape(a.shape)
    @classmethod
    def getitem_multi_index(cls, a: np.ndarray, ids: np.ndarray) -> np.ndarray:
        flat_a = a.flatten()
        flat_ids = np.ravel_multi_index(np.transpose(ids), a.shape)

        return flat_a[flat_ids]


    @classmethod
    def get_basis(cls, dims: tuple[int,int,int]) -> np.ndarray:
        return np.array([np.prod(dims[i+1:]) for i in np.arange(len(dims))])
    @classmethod
    def flatten_multi_index(cls, ids: np.ndarray, dims, dtype: np.dtype=None, axis: int=1) -> np.ndarray:
        '''
        # Notes

        Encodes a vector of n dimensions to scalar.
        There for the vector must be inside the hypercube given by dims parameter.
        An unique id is assigned to every position in this hypercube.
        The resulting scalar of a given vector is determined by the id of the cell of the hypercube the vector is pointing to.

        this concept is extended for tensors insted of vectors,
        by reshapeing the tensor and the dims to a 1d tensor.

        This implementation of the method only workes for vectors of integer values.

        # Examples

        >>> dims = (2,2)
        >>> vec_for_every_pos = [[0,0],[0,1],[1,0],[1,1]]
        >>> flatten_multi_index(vec_for_every_pos, dims)
        array([0., 1., 2., 3.])

        possible for n dims
        >>> ne.flatten_multi_index([[0,1,2],[1,2,3]], (4,4,4))
        array([ 6., 27.])

        encode a tensor
        >>> a = np.arange(8).reshape((2,2,2))
        >>> dims = (np.max(a),) *np.shape(a)[1] *np.shape(a)[2]
        >>> ne.flatten_multi_index(a, dims)
        array([  66, 1666])
        '''
        if dtype and type(ids) is np.ndarray:
            ids = ids.astype(dtype)
        elif dtype:
            ids = np.array(ids).astype(dtype)

        flatten_ids = cls.flatten(ids, axis=axis)
        flatten_dims = cls.flatten(dims)

        basis = cls.get_basis(flatten_dims)
        ids_transformed = np.multiply(flatten_ids, basis)
        return np.sum(ids_transformed, axis=1)


    @classmethod
    def one_hot(cls, values: np.ndarray, class_count: int=None) -> np.ndarray:
        """
        values must be in range [0, class_count)
        """
        if class_count is None:
            class_count = np.max(values)+1
        return np.eye(class_count)[np.reshape(values,-1)]

    @classmethod
    def flatten(cls, a: np.ndarray, axis: int=0) -> np.ndarray:
        shape = np.shape(a)
        new_shape = shape[:axis] + (np.prod(shape[axis:]),)
        return np.reshape(a, new_shape)

    @classmethod
    def getitem_nd(cls, a: np.ndarray, id_mtx: np.ndarray) -> np.ndarray:
        '''
        # Notes

        Makes it possible to generate view from multidimensional id-array.
        Therefor every id in id_mtx gets substituted by its related item in a.
        In the simplest case view_nd is equevalent to a[id].

        # Examples

        equevalent to a[id]
        >>> ne.view_nd([1,2,3], [2,0,1])
        array([3, 1, 2])
        
        multi dim id_mtx
        >>> id_mtx = np.array([[0,1,2], [1,2,0], [2,0,1]])
        >>> id_mtx
        array([[0, 1, 2],
                [1, 2, 0],
                [2, 0, 1]])
        >>> ne.view_nd([1,2,3], id_mtx)
        array([[1, 2, 3],
                [2, 3, 1],
                [3, 1, 2]])

        multi dim a and id_mtx
        >>> a = np.array([[1,2],[3,4],[5,6]])
        >>> a
        array([[1, 2],
                [3, 4],
                [5, 6]])
        >>> ne.view_nd(a, id_mtx)
        array([[[1, 2],
                [3, 4],
                [5, 6]],
               [[3, 4],
                [5, 6],
                [1, 2]],
               [[5, 6],
                [1, 2],
                [3, 4]]])
        '''
        if type(a) is not np.ndarray:
            a = np.array(a)
        if type(id_mtx) is not np.ndarray:
            id_mtx = np.array(id_mtx)

        ids_flatten = np.ravel(id_mtx)
        shape = (*id_mtx.shape, *(a.shape[1:])) if len(a.shape) > 1 else id_mtx.shape
        return a[ids_flatten].reshape(shape)

    @classmethod
    def setitem_nd(cls, a: np.ndarray, id_mtx: np.ndarray, values: np.ndarray) -> np.ndarray:
        if type(id_mtx) is not np.ndarray:
            id_mtx = np.array(id_mtx)
        if type(values) is not np.ndarray:
            values = np.array(values)

        ids_flatten = np.ravel(id_mtx)
        result = np.copy(a)
        shape = (np.prod(values.shape[:len(id_mtx.shape)]), *values.shape[len(id_mtx.shape):])
        result[ids_flatten] = np.reshape(values, shape)
        return result
    
    @classmethod
    def spectral_clustering(
        cls,
        laplacian_matrix: np.ndarray,
        k_clusters: int,
        k_eigenvectors: int=None) -> np.ndarray:
        if k_eigenvectors is None:
            k_eigenvectors = len(laplacian_matrix)

        eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

        sorted_ids = np.argsort(eigenvalues)
        embeddings = eigenvectors[sorted_ids][1:k_eigenvectors+1].T

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_
        return cluster_labels
    
    @classmethod
    def relabel(cls, a: np.ndarray) -> np.ndarray:
        """
        assigns ints to every value in a, such that same values in a get the same int starting by 0 for the first value in values.
        """
        _, unique_ids = np.unique(a, return_index=True)
        unique_ids_sorted = np.sort(unique_ids)

        values = a[unique_ids_sorted]
        replacements = np.arange(len(unique_ids_sorted))

        result = np.copy(a)
        shape = result.shape
        result = result.flatten()

        d = dict(zip(values, replacements))
        def r(x): return d[x]

        mask = np.in1d(result, values)
        result[mask] = np.frompyfunc(r, 1, 1)(result[mask])
        result = result.reshape(shape)
        return result.astype(replacements.dtype)
    
    @classmethod
    def extend_nd(cls, a: np.ndarray, dim: int) -> np.ndarray:
        """
        # Notes

        Extends the second axis of a to the given dimension.

        # Examples

        >>> ne.extend_nd(np.array([[1],[2],[3]]), 3)
        array([[1,0,0], [2,0,0], [3,0,0]])
        """

        missing_dims = max(dim - a.shape[1], 0)
        # pad with zeros on the right side of the second axis
        result_arr = np.pad(a, ((0, 0), (0, missing_dims)))
        return result_arr
    
    @classmethod
    def array_to_json(cls, a: np.ndarray) -> dict[str, any]:
        return {
            'data': a.tolist(),
            'dtype': str(a.dtype)
        }
    @classmethod
    def array_from_json(cls, d: dict[str, any]) -> np.ndarray:
        return np.array(d['data'], dtype=np.dtype(d['dtype']))
