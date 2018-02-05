
import numpy as np
from _TYPE_Grid import KerasQAgent

import time

start_time = time.time()
maze = np.zeros(6)
maze[5] = 10
agent = KerasQAgent(max_trades=1, position_bounds=[0, 5])
for _ in range(20000):
    position = np.random.randint(0, 6)
    action = agent.choose([position])
    agent.update(maze[position + action], [position], action, [position + action])

batch = np.array([[pos, act] for pos in range(6) for act in range(-1, 2)])
print(agent.q_map.predict(batch).reshape([6, 3]))

end_time = time.time()
print(end_time - start_time)


#
# feature_list = {
#     "level": {
#         "cells": 500,
#         "spec": []
#     }
# }
#
# process_type = ExampleOUProcess
# trader = UnivariateTabularTrader(feature_list, process_type())
# sim = Simulator(process_type(), trader)
# sim.train(int(5E6))
# sim.test(20000)
# print(sim.record)
# sim.record.plot()