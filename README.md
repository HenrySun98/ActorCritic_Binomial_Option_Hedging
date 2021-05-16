# ActorCritic_Binomial_Option_Hedging
One simple Python code to implement Actor Critic Algorithm for hedging European At-The-Money option under Binomial Tree model.

This is one of projects completed at HKUST ***MSc in Financial Mathematics (MAFM)*** program , 2021 Spring, for the course *MAFS6010Y Reinforcement Learning with Applications in Finance*.

**Reward Design** is closely related to the idea of hedge, NOT to make positive or negative PnL but to make your portfolio (one option + $\delta$ stocks) to have NO volatilities in value. Here I choose $-abs(R)$ as the reward for each steps of one episode.

**Binomial Tree** is the most simple option pricing techniques with emphasis on **replication**. One environment is created to represent this pricing idea. **Environment** is an important concept in reinforcement learning mainly to interact with the agent (i.e., the hedger) and offers **state & reward** update.
