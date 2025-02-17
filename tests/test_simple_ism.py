import unittest
from simple_ism import simple_ism

class TestSimpleISM(unittest.TestCase):
    def test_return_value(self):
        x = 0.1128  # Replace this with the value you expect the function to return
        self.assertEqual(simple_ism(create_subplots=False), x)

if __name__ == '__main__':
    unittest.main()
