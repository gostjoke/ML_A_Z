# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000  # total ad boradcast
d = 10     # total reward
ads_selected = []
numbers_of_selections = [0] * d #dp?
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0 
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            # before n roud,  Average round reward
            #' r(n) =    r/n
            average_reward = sums_of_rewards[i] / numbers_of_selections[i] 
            
            ### sigma(n)  
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            ### confidence [r(n)-delta_i, r(n)+delta_i]
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 ## 10**400, a very large number 
        if upper_bound > max_upper_bound:
            # now we find max upper bound
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 # next round so need to add 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    
    ### total reward is more than random select, we gets more reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()