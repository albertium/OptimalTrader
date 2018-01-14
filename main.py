
import numpy as np

from OptimalTrader import GridTrader
from StochasticProcess import ExampleOUProcess
from Grid import ActionValueGrid


# grid = ActionValueGrid()
# grid.add_states({"name": "price", "state_type": "continuous"})

a = np.arange(10)

def get(input):
    return input[1]

lv = get(a)
lv = 100
print(a)





# process = ExampleOUProcess()
# trader = GridTrader(process)
# trader.initialize_grid(100, 0, 10, 0, 10000, 100)
# print(trader.process_generate(10))

