import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm

warnings.filterwarnings('ignore')


class BinomialTree:
    def __init__(self, s0=100, up_prob=0.5, up_pct=1.2, risk_free=0.02, time=1, maturity=20):
        # dynamics setting
        self.init_price = s0
        self.s0 = s0
        self.up_prob = up_prob
        self.down_prob = 1 - self.up_prob
        self.up_pct = up_pct
        self.down_pct = 1 / self.up_pct
        self.risk_free = risk_free
        self.t = time
        self.maturity = maturity

        self.option_pos = 1.0
        self.stock_pos = None

    def reset(self):
        self.s0 = self.init_price

    def get_ATM_strike(self):
        return self.s0 * (1 + self.risk_free) ** self.t

    def risk_neutral(self):
        return (1 + self.risk_free - self.down_pct) / (self.up_pct - self.down_pct)

    def get_option_val(self):
        up_price = self.s0 * self.up_pct
        down_price = self.s0 * self.down_pct
        prob = self.risk_neutral()
        strike = self.get_ATM_strike()
        c_up = max(0, up_price - strike)
        c_down = max(0, down_price - strike)
        option_val = (c_up * prob + c_down * (1 - prob)) / (1 + self.risk_free)
        return option_val

    def step(self, action):
        """

        :param action: hedge amount ( continuous number )
        :return: new_observation, reward
        """
        stock_val1 = self.s0
        option_val1 = self.get_option_val()

        if np.random.binomial(1, self.up_prob) == 1:
            self.s0 *= self.up_pct
        else:
            self.s0 *= self.down_pct

        stock_val2 = self.s0
        option_val2 = self.get_option_val()

        reward = self.reward(option_val_change=option_val2/(option_val1+1e-3)-1,
                             stock_val_change=stock_val2/(stock_val1+1e-3)-1,
                             hedge_amount=action)

        return self.s0, reward

    def reward(self, option_val_change, stock_val_change, hedge_amount):
        delta_portfolio = self.option_pos * option_val_change - hedge_amount * stock_val_change
        reward = -abs(delta_portfolio)
        return reward


# -------------------------------- Actor ==> actions -------------------------------- #
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, action_dim)
        self.linear3.weight.data.uniform_(-0.03, 0.03)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = self.linear3(x)
        return action


# -------------------------------- Critic ==> Value Function Q(S, a) -------------------------------- #
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(action_dim, 32)
        self.linear3 = nn.Linear(32, 1)
        self.linear3.weight.data.uniform_(-0.03, 0.03)

    def forward(self, state, action):
        x = F.relu(self.linear1(state))
        y = F.relu(self.linear2(action))
        value = torch.stack((x, y))
        value = self.linear3(value)
        return value[0]


class Trainer:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.iter = 0
        self.len = 0

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state)
        new_action = action.data.numpy()
        return F.sigmoid(new_action)

    def _step(self, state):
        state = torch.FloatTensor(np.float32(state))
        action = self.actor.forward(state)
        value_now = torch.squeeze(self.critic.forward(action=action, state=state))
        new_state, reward = env.step(action)
        new_state = torch.FloatTensor(np.float32(torch.from_numpy(np.array([new_state]))))
        value_next = torch.squeeze(self.critic.forward(action=action, state=new_state))
        # ---------------------- Calculate TD-Error / Advantage ---------------------- #
        td_error = reward + GAMMA * value_next - value_now
        return state, action, reward, new_state, td_error

    def optimize(self, state):
        # ---------------------- Optimize Critic : MSE Loss ---------------------- #
        state, action, reward, new_state, td_error = self._step(state)
        y_predicted = torch.squeeze(self.critic.forward(state=state, action=action))
        loss_critic = F.smooth_l1_loss(y_predicted, td_error)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- Optimize Actor---------------------- #
        state, action, reward, new_state, td_error = self._step(state)
        m = Categorical(action)
        action_prob = m.sample()
        loss_actor = m.log_prob(action_prob) * td_error
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        return action, new_state, reward


def delta_hedging(stock_price, up_pct, risk_free, time=1, maturity=20):
    up_price = stock_price * up_pct
    down_price = stock_price / up_pct
    atm_k = stock_price * (1+risk_free) ** time
    delta = (max(up_price-atm_k, 0) - max(down_price-atm_k, 0)) / (up_price-down_price)
    return delta


def cal_hedging_performance(stock_path, hedge_amount, risk_free=0.02, time=1, maturity=20):
    option_val_arr = []
    k = stock_path[0] * (1 + risk_free) ** maturity
    for i in range(len(stock_path)-1):
        option_val = max(stock_path[i+1] - k, 0)
        option_val_arr.append(option_val)
    stock_pos = [i * j for i, j in zip(hedge_amount[2:], stock_path[3:])]
    option_pos = [i-j for i, j in zip(option_val_arr[2:], option_val_arr[1:-1])]
    performance = [i+j for i, j in zip(stock_pos, option_pos)]
    reward = [-abs(i) for i in performance]
    return reward


if __name__ == "__main__":

    # --------------------------- Environment Params --------------------------- #
    S0 = 100
    UP_PROB = 0.4
    UP_PCT = 1.2
    RISK_FREE_RATE = 0.01
    TIME = 1
    MATURITY = 10

    env = BinomialTree(s0=S0, up_prob=UP_PROB, up_pct=UP_PCT, risk_free=RISK_FREE_RATE, time=TIME, maturity=MATURITY)

    # --------------------------- Training Hyper Params --------------------------- #
    BATCH_SIZE = 32
    LEARNING_RATE = 0.003
    GAMMA = 1-RISK_FREE_RATE
    MAX_EPISODES = 1000
    MAX_STEPS = MATURITY

    trainer = Trainer(state_dim=1, action_dim=1)

    # --------------------------- Collecting Results --------------------------- #
    collect_action = False
    collect_path = False
    AC_action_arr = None
    DELTA_action_arr = None
    Path_arr = None

    # --------------------------- Start to Train --------------------------- #
    episode_reward = []
    for _ep in tqdm(range(MAX_EPISODES)):
        # if (_ep+1) % round(MAX_EPISODES/20) == 0:
        #     print('Episode ', _ep+1, ' / ', MAX_EPISODES)
        env.reset()
        reward_arr = []
        observation = np.array([env.s0])

        if _ep == MAX_EPISODES - 1:
            collect_action = True
            collect_path = True
            AC_action_arr = []
            DELTA_action_arr = []
            Path_arr = []

        for r in range(MAX_STEPS):
            if r == 0:
                observation = Variable(torch.from_numpy(observation))
            # perform optimization
            action, new_state, reward = trainer.optimize(state=observation)
            reward_arr.append(reward.detach().numpy())
            observation = new_state

            # collect action
            if collect_action:
                observation = observation.detach().numpy()
                AC_action_arr.append(action)
                delta = delta_hedging(stock_price=observation, up_pct=UP_PCT, risk_free=RISK_FREE_RATE, time=TIME)
                DELTA_action_arr.append(delta)
                Path_arr.append(float(observation))

        episode_reward.append(np.mean(reward_arr))

    # --------------------------- Plot the results --------------------------- #
    # Mean reward
    plt.plot(episode_reward, lw=3)
    plt.title('Mean Reward per Episode', fontsize=30)
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Mean Reward', fontsize=20)
    plt.show()
    # Action
    plt.plot(AC_action_arr, lw=2, label='Actor_Critic Hedge')
    plt.plot(DELTA_action_arr, lw=2, label='Delta Hedge')
    plt.title('Hedge Decision', fontsize=30)
    plt.xlabel('TIME', fontsize=20)
    plt.ylabel('Hedge Amount', fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    # hedging performance
    ac_perf = cal_hedging_performance(stock_path=Path_arr, hedge_amount=AC_action_arr,
                                      risk_free=RISK_FREE_RATE, time=TIME, maturity=MATURITY)
    delta_perf = cal_hedging_performance(stock_path=Path_arr, hedge_amount=DELTA_action_arr,
                                         risk_free=RISK_FREE_RATE, time=TIME, maturity=MATURITY)
    plt.plot(ac_perf, lw=2, label='Actor_Critic Reward')
    plt.plot(delta_perf, lw=2, label='Delta Hedge Reward')
    plt.title(r'Hedge Reward : $-abs(R)$', fontsize=30)
    plt.xlabel('TIME', fontsize=20)
    plt.ylabel('Hedge Performance', fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
