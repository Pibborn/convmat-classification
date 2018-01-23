from extract_conv_mat import is_even, is_odd
import unittest
import numpy as np

class TestSymCheckFunctions(unittest.TestCase):

   def test_even(self):
       a = np.array([[1, 5, 1], [5, 5, 5], [1, 5, 1]])
       self.assertTrue(is_even(a, type='HORIZONTAL'))
       self.assertTrue(is_even(a, type='VERTICAL'))
       self.assertTrue(is_even(a, type='DIAGSX'))
       self.assertTrue(is_even(a, type='DIAGDX'))

   def test_odd(self):
       a = np.array([[1, 5, 1], [-5, 5, 5], [-1,-5,-1]])
       self.assertTrue(is_odd(a, type='HORIZONTAL'))
       self.assertTrue(is_odd(a, type='DIAGSX'))

       a = np.array([[1, 2, -1], [2, 0, -2], [1, -2, -1]])
       self.assertTrue(is_odd(a, type='DIAGDX'))
       self.assertTrue(is_odd(a, type='VERTICAL'))

if __name__ == '__main__':
    unittest.main()
