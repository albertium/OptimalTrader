
import numpy as np
import time
import plotly.offline as offline
import plotly.graph_objs as go

from _LIB_Core import plot_lines


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

    def train(self, n_epochs=100000):
        self._initialize_trader_features()

        q_record = []
        dw_record = 0
        curr_price = self.generator.generate()
        start_time = time.time()
        curr_position = 0
        for epoch in range(n_epochs):
            new_position = self.trader.trade(curr_price, curr_position)
            new_price = self.generator.generate()
            dw = new_position * (new_price - curr_price) - \
                 (self.commission + self.spread) * np.abs(new_position - curr_position)
            reward = dw - 0.5 * self.kappa * dw * dw
            dw_record += dw
            self.trader.update(new_price, new_position, reward)
            curr_price = new_price
            curr_position = new_position

            if (epoch + 1) % 1000 == 0:
                q_value = self.trader.get_average_q_value()
                print("\rTraining [%d%%] ... %f (q value) / %f (pnl)" %
                      (epoch / n_epochs * 100, q_value, dw_record), end="", flush=True)
                q_record.append(q_value)
                dw_record = 0
        print()
        end_time = time.time()
        print("Training used time:  %.3f s" % (end_time - start_time))
        plot_lines({"q value": q_record}, "Convergence")

    def test(self, n=10000):
        self._initialize_trader_features()

        curr_price = self.generator.generate()
        curr_position = 0
        record = np.zeros(n)
        cost = np.zeros(n)
        for epoch in range(n):
            new_position = self.trader.trade(curr_price, curr_position)
            new_price = self.generator.generate()
            record[epoch] = new_position * (new_price - curr_price)
            cost[epoch] = -(self.commission + self.spread) * np.abs(new_position - curr_position)

            curr_price = new_price
            curr_position = new_position

        self.record = DetailPerformance(record, cost)

    def _initialize_trader_features(self) -> None:
        if self.trader.pre_run_lags > 0:
            burn_in = self.generator.generate(self.trader.pre_run_lags)
            if isinstance(burn_in, float):
                burn_in = [burn_in]
            self.trader.initialize_features(burn_in)


class Performance:
    def __init__(self, record):
        self.record = np.array(record)

        # calculate metrics
        self.gross_profit = float(np.sum(np.maximum(0, self.record)))
        self.gross_loss = float(np.sum(np.minimum(0, record)))
        self.total_pnl = np.sum(record)
        self.percent_win = float(np.sum(record > 0) / len(record))
        max_dd = 0
        running = 0
        running_sum = 0
        local_peak = 0
        peak = 1
        for ret in self.record:
            running_sum += ret
            running = min(0, running + ret)
            if running_sum > local_peak:
                local_peak = running_sum
            if running < max_dd:
                max_dd = running
                peak = local_peak
        self.max_drawdown = max_dd
        if peak == 0:
            self.peak = 1
        else:
            self.peak = peak
        self.percent_drawdown = -self.max_drawdown / self.peak

    def __str__(self):
        to_print = "Total PnL: %.3f\n" % (self.gross_profit + self.gross_loss)
        to_print += "Gross win/loss: %.3f/%.3f\n" % (self.gross_profit, self.gross_loss)
        to_print += "Percent winning: %.1f%%\n" % (self.percent_win * 100)
        to_print += "Max drawdown: %.3f (%.2f%%)\n" % (self.max_drawdown, self.percent_drawdown * 100)
        return to_print

    def plot(self):
        trace = go.Scatter(y=np.cumsum(self.record), mode="lines")
        offline.plot([trace])

    def get_performance(self):
        return {
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "percent_win": self.percent_win,
            "max_drawdown": self.max_drawdown,
            "percent_drawdown": self.percent_drawdown
        }


class DetailPerformance:
    def __init__(self, record, cost):
        record, cost = np.array(record), np.array(cost)
        self.total_record = Performance(record + cost)
        self.record_only = Performance(record)
        self.total_pnl = self.total_record.total_pnl
        self.percent_win = self.total_record.percent_win

    def __str__(self):
        to_print = "================  Total ================\n"
        to_print += self.total_record.__str__()
        to_print += "\n\n================  w/o Cost ================\n"
        to_print += self.record_only.__str__()
        return to_print

    def plot(self) -> None:
        plot_lines({
            "PnL": np.cumsum(self.total_record.record),
            "w/o cost": np.cumsum(self.record_only.record)
        }, "Performance")
