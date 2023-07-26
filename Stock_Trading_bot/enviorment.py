import gym
import gym_anytrading
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt

class Actions(Enum):
    Sell = 0
    Hold = 1
    Buy = 2


class Positions(Enum):
    Short = 0
    Hold = 1
    Long = 2


def opposite(position: Positions, action: int) -> any:
    '''New opposite, so that it works with 3 actions, overwiev:
            When action is Sell, we either Go to Hold (when we bought earlier) or short (when position was flat before)

            When action is Buy, we conduct the same reasoning but in the oppiste way

            When the action is Hold we stay at current position

    '''

    action = Actions(action)
    if action == Actions.Sell:

        if position == Positions.Long:
            return Positions.Hold, False

        if position == Positions.Hold:
            return Positions.Short, True

    if action == Actions.Buy:
        if position == Positions.Short:
            return Positions.Hold, False

        if position == Positions.Hold:
            return Positions.Long, True

    return position, False


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Hold
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.trade = False
        self.history = {}
        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        self._position, self.trade = opposite(self._position, action)
        if self.trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            elif position == Positions.Hold:
                color = 'blue'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        hold_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            elif self._position_history[i] == Positions.Hold:
                hold_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError


# from .trading_env import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, bid_percent = 0.0003, leverge = 1):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)
        self.trade_fee_bid_percent = bid_percent  # unit
        self.trade_fee_ask_percent = bid_percent
        self.leverge = leverge
        # # unit

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'Close'].to_numpy()[start:end]
        print(prices.shape)
        signal_features = self.df.iloc[:, ~self.df.columns.isin(['Date', 'Close'])].to_numpy()[start:end]
        normalized_df = (signal_features - signal_features.min()) / (signal_features.max() - signal_features.min())

        signal_features = normalized_df
        return prices, signal_features

    #     def _process_data(self):
    #         prices = self.df.loc[:, 'Close'].to_numpy()

    #         prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
    #         prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

    #         diff = np.insert(np.diff(prices), 0, 0)
    #         signal_features = np.column_stack((prices, diff))

    #         return prices, signal_features
    #
    # def _calculate_reward(self, action):
    #     step_reward = 0
    #
    #     if self.trade:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]
    #         price_diff = current_price - last_trade_price
    #
    #         if self._position == Positions.Long:
    #             step_reward += price_diff
    #         elif self._position == Positions.Short:
    #             step_reward += -price_diff
    #
    #     return step_reward

# This reward function is specified so that it'll consider bid and ask costs
    def _calculate_reward(self, action):
        step_reward = 0
        if self.trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            # price_diff = current_price - last_trade_price
# there was an error here, short position worked the other way it should KEKW
            if self._position == Positions.Long:
                step_reward += self.leverge*current_price*(1-self.trade_fee_bid_percent) - last_trade_price*(1+self.trade_fee_ask_percent)
            elif self._position == Positions.Short:
                step_reward += -self.leverge*current_price*(1+self.trade_fee_bid_percent) + last_trade_price*(1-self.trade_fee_ask_percent)

        return step_reward
    # Is this actually correct? reutrn for short shuld be different I suppouse

    def _update_profit(self, action):

        if self.trade or self._done:
            current_price = self.leverge * self.prices[self._current_tick]
            last_trade_price = self.leverge * self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
            elif self._position == Positions.Short:
                self._total_profit = (self._total_profit * (1 + (
                            last_trade_price - current_price - self.trade_fee_bid_percent * last_trade_price - self.trade_fee_ask_percent * current_price) / last_trade_price))

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return