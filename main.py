
import numpy as np

from OptimalTrader import GridTrader
from StochasticProcess import ExampleOUProcess
from Grid import ActionValueGrid


# grid = ActionValueGrid()
# grid.add_states({"name": "price", "state_type": "continuous"})

a = np.array([np.arange(10) * i for i in range(10)])
print(a)
index = [[1, 8], [1, 2, 3]]
print(a[np.ix_(*index)])

print(not [])


# process = ExampleOUProcess()
# process.plot(1000)
# trader = GridTrader(process)
# trader.initialize_grid(100, 0, 10, 0, 10000, 100)
# print(trader.process_generate(10))

