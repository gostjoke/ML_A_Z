# Thompson Sampling
"""
湯普森抽樣（Thompson sampling）是一种用于多臂赌博机问题（multi-armed bandit problem）的随机化策略。
多臂赌博机问题是一个经典的强化学习问题，其中有多个“臂”（也称为“老虎机”或“赌博机”），每个臂代表一个可选的行动
，执行该行动可能会产生奖励。每个臂的奖励分布通常是未知的，玩家的目标是在有限的时间内最大化累积奖励。

湯普森抽樣的核心思想是将每个臂的奖励分布建模为一个概率分布，并使用贝叶斯统计方法来不断更新这些分布
。具体来说，湯普森抽樣将每个臂的奖励分布建模为Beta分布，
其中Beta分布是一个用于表示0到1之间的随机变量的概率分布。初始时，对于每个臂，
我们可以将Beta分布的参数设置为先验分布，通常选择均匀分布（即先验分布的参数alpha和beta都为1）。

然后，在每个时间步中，湯普森抽樣执行以下步骤：

对于每个臂，从其Beta分布中随机采样一个值。这些采样值代表了对该臂奖励分布的当前估计。

选择具有最高采样值的臂作为当前时间步的动作。

执行所选动作，并观察获得的奖励。

根据观察到的奖励，更新对每个臂的奖励分布的Beta分布参数。这是通过将观察到的奖励添加到先前的参数中来实现的。

通过不断更新奖励分布，湯普森抽樣可以自适应地选择在过去表现良好的臂，同时保持对未知臂的探索。
这种方法具有理论上的优势，并已被广泛用于在线广告投放、医疗试验设计和资源分配等领域。

总之，湯普森抽樣是一种用于解决多臂赌博机问题的策略，通
过不断更新奖励分布的贝叶斯方法来实现有效的探索和利用，以最大化累积奖励。
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d # the number which award to AD , award is 1
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        # ad only can be 1 or 0, means reward or not get reward
        # get highest random beta
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()