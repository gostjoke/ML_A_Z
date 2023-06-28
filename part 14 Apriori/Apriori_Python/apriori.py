# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
# generator
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
### if we set min_confidence too high, some relation is not reasonable


# Visualising the results
results = list(rules)
myResults = [list(x) for x in results]
# console myResults[:]
"""
[frozenset({'chicken', 'light cream'}),
 0.004532728969470737,
 [OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), 
                   confidence=0.29059829059829057, lift=4.84395061728395)]]

支持度（Support）：0.004532728969470737。這表示在數據集中，同時包含 "chicken" 和 "light cream" 的交易數量佔
總交易數量的比例為 0.004532728969470737。

關聯規則（Association Rule）：該規則的左側（antecedent）是 "light cream"，右側（consequent）是 "chicken"。
該規則的統計信息如下：

信心（Confidence）：0.29059829059829057。這表示在所有包含 "light cream" 的交易中，約有 29.06% 的交易也包含
 "chicken"。提升度（Lift）：4.84395061728395。提升度指的是左側和右側商品之間的關聯性。
 這個值大於 1 表示左側和右側是正相關的，而提升度越高表示關聯性越強。
綜合起來，這個結果顯示在數據集中購買 "light cream" 的客戶也有一定的機會購買 "chicken"，
並且這種關聯性比隨機情況下的購買機會要高 4.84 倍。

"""
    