import unittest
from simple_ntf import simple_ntf

class TestSimpleNTF(unittest.TestCase):
    def test_return_value(self):
        x = 0.2631  # Replace this with the value you expect the function to return
        self.assertEqual(simple_ntf(create_subplots=False), x)

if __name__ == '__main__':
    unittest.main()
