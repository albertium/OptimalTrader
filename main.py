

import numpy as np
from _TYPE_OptimalTrader import UnivariateGridTrader
from _TYPE_StochasticProcess import ExampleOUProcess


trader = UnivariateGridTrader()
process = ExampleOUProcess()
trader.add_process(process)
# add features
features = {
    "level": {
        "cells": 500,
        "spec": []
    }
}
trader.add_features(features)
trader.train(100000)
trader.q_table.epsilon = 1
trader.test()
