import unittest
import utils

sigma = 25
x = 3

class TestCalc(unittest.TestCase):

    def test_gaussian(self):
        self.assertEqual(utils.gaussian(x, sigma), 0.015843208471746244)
    
    def test_gaussian1d(self):
        self.assertEqual(utils.gaussian1d(x, sigma), -7.604740066438197e-05)

    def test_gaussian2d(self):
        self.assertEqual(utils.gaussian2d(x, sigma), -2.4984106031604957e-05)

    def test_computegaussian(self):
        g, g1, g2 = utils.compute_gaussian(x, sigma)

        self.assertEqual(g, 0.015843208471746244)
        self.assertEqual(g1, -7.604740066438197e-05)
        self.assertEqual(g2, -2.4984106031604957e-05)

if __name__ == '__main__':
    unittest.main()