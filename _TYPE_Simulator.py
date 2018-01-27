
import numpy as np
import plotly.offline as offline
import plotly.graph_objs as go


class Simulator:
    def __init__(self, generator, trader, params=None):
        self.generator = generator
        self.trader = trader
        self.record = None

        if not params:
            params = {}
        self.commission = params.get("commission", 0.1)
        self.spread = params.get("spread", 0.1)
        self.lot_size = params.get("lot_size", 100)
        self.kappa = params.get("kappa", 1E-4)

    def train(self, n=100000):
        curr_price = self.generator.generate()
        for _ in range(n):
            prev_position, actions = self.trader.choose(curr_price)  # this allows update multiple actions
            curr_positions = prev_position + actions
            next_price = self.generator.generate()
            dw = curr_positions * (next_price - curr_price) - (self.commission + self.spread) * np.abs(actions)
            reward = dw - 0.5 * self.kappa * dw * dw
            self.trader.update(reward)
            curr_price = next_price

    def test(self, n=10000):
        curr_price = self.generator.generate()
        prev_position = 0
        record = np.zeros(n)
        for epoch in range(n):
            curr_position = self.trader.trade(curr_price)
            next_price = self.generator.generate()
            dw = curr_position * (next_price - curr_price) \
                - (self.commission + self.spread) * np.abs(curr_position - prev_position)
            record[epoch] = dw

            curr_price = next_price
            prev_position = curr_position

        self.record = Performance(record)


class Performance:
    def __init__(self, record):
        self.record = np.array(record)

        # calculate metrics
        self.gross_profit = float(np.sum(np.maximum(0, self.record)))
        self.gross_loss = float(np.sum(np.minimum(0, record)))
        self.percent_win = float(np.sum(record > 0) / len(record))
        max_dd = 0
        running = 0
        for ret in self.record:
            running = min(0, running + ret)
            max_dd = min(max_dd, running)
        self.max_drawdown = max_dd

    def __str__(self):
        to_print = "Total PnL: %.3f\n" % (self.gross_profit + self.gross_loss)
        to_print += "Gross win/loss: %.3f/%.3f\n" % (self.gross_profit, self.gross_loss)
        to_print += "Percent winning: %.1f%%\n" % (self.percent_win * 100)
        to_print += "Max drawdown: %.3f\n" % self.max_drawdown
        return to_print

    def plot(self):
        trace = go.Scatter(y=np.cumsum(self.record), mode="lines")
        offline.plot([trace])

    def get_performance(self):
        return {
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "percent_win": self.percent_win,
            "max_drawdown": self.max_drawdown
        }