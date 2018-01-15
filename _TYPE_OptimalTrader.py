
import numpy as np
import abc


class OptimalTrader:
    def __init__(self, process=None, max_trade=5, max_position=10, lot_size=100):
        self.process = process
        self.process_generate = None  # for re-scale process
        self.max_trade = max_trade
        self.max_position = max_position
        self.lot_size = lot_size

    @abc.abstractclassmethod
    def train(self):
        pass

    @staticmethod
    def rescale_univariate(y_max, y_min, num_rows):
        const = 1 / (y_max - y_min) * num_rows

        def wrapper(generate_func):
            def rescaled(*args):
                results = np.array(generate_func(*args))
                return np.floor((results - y_min) * const).astype(int)
            return rescaled
        return wrapper


class GridTrader(OptimalTrader):
    def __init__(self, process=None, max_trade=5, max_position=10, lot_size=100):
        super().__init__(process, max_trade, max_position, lot_size)
        self.grid = None
        self.grid_spec = {"y_max": np.inf, "y_min": 0, "x_max": np.inf, "x_min": -np.inf}

    def initialize_grid(self, y_max, y_min, x_max, x_min, num_rows, num_cols):
        """
        Set up grid and re-sscaled process
        """

        # set up grid
        self.grid_spec = {"y_max": y_max, "y_min": y_min, "x_max": x_max, "x_min": x_min}
        self.grid = np.zeros((num_rows, num_cols))

        # create re-scaled process
        @OptimalTrader.rescale_univariate(y_max, y_min, num_rows)
        def generate_func(*args):
            return self.process.generate(*args)

        self.process_generate = generate_func

    def train(self):
        pre_run = self.process.generate(1000)
        _min, _max = np.min(pre_run), np.max(pre_run)
        # self.initialize_grid(_max * 1.2, _min / 1.2, self.)



