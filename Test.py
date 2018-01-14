
import unittest
import numpy as np

from StochasticProcess import ExampleOUProcess
from OptimalTrader import GridTrader


class OptimalTraderTestCase(unittest.TestCase):
    """
    test OptimalTrader.py
    """

    def test_if_integer(self):
        process = ExampleOUProcess()
        trader = GridTrader(process)
        trader.initialize_grid(100, 0, 10, -10, 1000, 10000)
        self.assertTrue(isinstance(trader.process_generate(), np.int64))
        self.assertTrue(isinstance(trader.process_generate(10), np.ndarray))


if __name__ == "__main__":
    unittest.main()