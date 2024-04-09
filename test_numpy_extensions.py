import unittest
from numpy_extensions import numpy_extensions as npe
import numpy as np

class TestCaseNp(unittest.TestCase):
    def assertArrayEqual(self, first, second):
        result = np.array_equal(first, second)
        msg = '' if result else f'arrays are different\n{first}\n!=\n{second}'
        self.assertTrue(result, msg)

class TestNumpyExtensions(TestCaseNp):
    @classmethod
    def setUpClass(cls):
        # Initialize test arrays
        cls.arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        cls.arr2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    # def test_flatten_default_axis(self):
    #     flattened_arr1 = npe.flatten(self.arr1)
    #     flattened_arr2 = npe.flatten(self.arr2)
        
    #     self.assertEqual(flattened_arr1.shape, (6,))
    #     self.assertEqual(flattened_arr2.shape, (8,))
    #     # self.assertEqual(flattened_arr2.shape, (2, 4))
    # def test_flatten_axis_0(self):
    #     flattened_arr1 = npe.flatten(self.arr1, axis=0)
    #     flattened_arr2 = npe.flatten(self.arr2, axis=0)
        
    #     self.assertEqual(flattened_arr1.shape, (6, ))
    #     self.assertEqual(flattened_arr2.shape, (8, ))
    # def test_flatten_axis_1(self):
    #     flattened_arr1 = npe.flatten(self.arr1, axis=1)
    #     flattened_arr2 = npe.flatten(self.arr2, axis=1)
        
    #     self.assertEqual(flattened_arr1.shape, (2, 3))
    #     self.assertEqual(flattened_arr2.shape, (2, 4))

    def test_contains_1d(self):
        a = [0,1,2]
        b = [0,3,1,4]
        self.assertArrayEqual(npe.contains(a,b), [True, False, True, False])
    def test_contains_2d_2(self):
        a = [[0,1],[1,2]]
        b = [[0,1],[0,2],[1,2]]
        self.assertArrayEqual(npe.contains(a,b), [True, False, True])
    def test_contains_2d_3(self):
        a = [[0,1,2],[1,2,3]]
        b = [[0,1,2],[0,2,4],[1,2,3]]
        self.assertArrayEqual(npe.contains(a,b), [True, False, True])
    def test_contains_3d_2(self):
        a = np.arange(8).reshape((2,2,2))
        b = [[[0,1],[2,3]], [[4,5],[6,0]]]
        self.assertArrayEqual(npe.contains(a,b), [True, False])

    def test_argcontains(self): pass

    def test_replace_single_vals(self):
        a = [1,2,3,2,1]
        expected = [0,2,3,2,0]
        self.assertArrayEqual(npe.replace(a, 1, 0), expected)
    def test_replace_multi_vals(self):
        a = [1,2,3,2,1]
        expected = [-1,-2,3,-2,-1]
        self.assertArrayEqual(npe.replace(a, [1,2], [-1,-2]), expected)
    def test_replace_multi_dims(self):
        a = [[1,2],[1,3],[2,4]]
        expected = [[0, 1],[0, 2],[1, 3]]
        self.assertArrayEqual(npe.replace(a, [1,2,3,4], [0,1,2,3]), expected)

    def test_getitem_nd_1dim_a_1dim_id(self):
        a = [1,2,3]
        id_mtx = [2,0,1]
        expected = [3,1,2]
        self.assertArrayEqual(npe.getitem_nd(a, id_mtx), expected)
    def test_getitem_nd_1dim_a_2dim_id(self):
        a = [1,2,3]
        id_mtx = [[0,1,2],[1,2,0],[2,0,1]]
        expected = [[1,2,3],[2,3,1],[3,1,2]]
        self.assertArrayEqual(npe.getitem_nd(a, id_mtx), expected)
    def test_getitem_nd_2dim_a_1dim_id(self):
        a = [[1,2],[3,4],[5,6]]
        id_mtx = [2,0,1]
        expected = [[5,6],[1,2],[3,4]]
        self.assertArrayEqual(npe.getitem_nd(a, id_mtx), expected)
    def test_getitem_nd_2dim_a_2dim_id(self):
        a = [[1,2],[3,4],[5,6]]
        id_mtx = [[0,1,2],[1,2,0],[2,0,1]]
        expected = [
            [[1,2],[3,4],[5,6]],
            [[3,4],[5,6],[1,2]],
            [[5,6],[1,2],[3,4]]
        ]
        self.assertArrayEqual(npe.getitem_nd(a, id_mtx), expected)

    def test_setitem_nd_1dim_a_1dim_id(self):
        a = [1,2,3]
        id_mtx = [2,0,1]
        values = [3,4,5]
        expected = [4,5,3]
        self.assertArrayEqual(npe.setitem_nd(a, id_mtx, values), expected)
    # def test_setitem_nd_zero_mtx(self):
    #     a = np.zeros((3, 2))
    #     id_mtx = np.array([0, 2])
    #     values = np.array([[10, 20], [30, 40]])
    #     expected = [4,5,3]
    #     print(npe.setitem_nd(a, id_mtx, values))
    #     self.assertArrayEqual(npe.setitem_nd(a, id_mtx, values), expected)
    def test_setitem_nd_1dim_a_2dim_id(self):
        a = [1,2,3,4]
        id_mtx = [[0,1],[2,3]]
        values = [[5,6],[7,8]]
        expected = [5,6,7,8]
        self.assertArrayEqual(npe.setitem_nd(a, id_mtx, values), expected)
    def test_setitem_nd_2dim_a_1dim_id(self):
        a = [[1,2],[3,4],[5,6]]
        id_mtx = [0,2]
        values = [[7,8],[9,0]]
        expected = [[7,8],[3,4],[9,0]]
        self.assertArrayEqual(npe.setitem_nd(a, id_mtx, values), expected)
    def test_setitem_nd_2dim_a_2dim_id(self):
        a = [[1,2],[3,4],[5,6],[7,8]]
        id_mtx = [[0,1],[2,3]]
        values = [[[-1,-2],[-3,-4]], [[-5,-6],[-7,-8]]]
        expected = [[-1,-2],[-3,-4],[-5,-6],[-7,-8]]
        self.assertArrayEqual(npe.setitem_nd(a, id_mtx, values), expected)

    def test_one_hot_single_value(self):
        value = np.array([0])
        expected_result = np.array([[1.]])
        result = npe.one_hot(value)
        np.testing.assert_array_equal(result, expected_result)
    def test_one_hot_single_value_class_count(self):
        values = np.array([0])
        class_count = 3
        expected_result = np.array([[1, 0, 0]])
        assert np.array_equal(npe.one_hot(values, class_count), expected_result)
    def test_one_hot_default_class_count(self):
        values = np.array([0, 1, 2, 0, 1])
        expected_result = np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.],
                                    [1., 0., 0.],
                                    [0., 1., 0.]])
        result = npe.one_hot(values)
        np.testing.assert_array_equal(result, expected_result)
    def test_one_hot_custom_class_count(self):
        values = np.array([1, 2, 1, 0, 0])
        class_count = 4
        expected_result = np.array([[0., 1., 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 1., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [1., 0., 0., 0.]])
        result = npe.one_hot(values, class_count=class_count)
        np.testing.assert_array_equal(result, expected_result)

    def test_relabel_basic(self):
        a = np.array([3, 1, 2, 1, 3, 2, 4])
        expected_result = np.array([0, 1, 2, 1, 0, 2, 3])
        result = npe.relabel(a)
        np.testing.assert_array_equal(result, expected_result)
    def test_relabel_negative_values(self):
        a = np.array([-3, -1, 2, 1, -3, 2, 4])
        expected_result = np.array([0, 1, 2, 3, 0, 2, 4])
        result = npe.relabel(a)
        np.testing.assert_array_equal(result, expected_result)
    def test_relabel_float(self):
        a = np.array([3.5, 1.2, 2.8, 1.2, 3.5, 2.8, 4.3])
        expected_result = np.array([0, 1, 2, 1, 0, 2, 3])
        result = npe.relabel(a)
        np.testing.assert_array_equal(result, expected_result)
    def test_relabel_string(self):
        a = np.array(['a', 'bb', 'ccc', 'bb', 'a', 'ccc', 'dddd'])
        expected_result = np.array([0, 1, 2, 1, 0, 2, 3])
        result = npe.relabel(a)
        np.testing.assert_array_equal(result, expected_result)

    def test_extend_nd_1d(self):
        input_arr_1d = np.array([[1], [2], [3]])
        expected_output_1d = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        self.assertTrue(np.array_equal(npe.extend_nd(input_arr_1d, 3), expected_output_1d))
    def test_extend_nd_2d(self):
        input_arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
        expected_output_2d = np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0]])
        self.assertTrue(np.array_equal(npe.extend_nd(input_arr_2d, 3), expected_output_2d))
    def test_extend_nd_3d(self):
        input_arr_3d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        expected_output_3d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertTrue(np.array_equal(npe.extend_nd(input_arr_3d, 3), expected_output_3d))


if __name__ == '__main__':
    unittest.main()