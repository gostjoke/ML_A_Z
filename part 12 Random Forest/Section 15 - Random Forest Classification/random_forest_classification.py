# Random Forest Classification
"""
1.Pick at random K data points from Training set

2. build The Decision Tree associated to theese K data points

3. choose the number Ntree of trees you want to build and repeat in steps 1,2

4. for a new data point, make each one of your Ntree trees predict the category
to which the data points beling, and assign the new data point to the category that 
wins the majority votes.
随机森林算法的优点包括：

可以处理高维数据和大规模数据集。
具有较高的准确性和鲁棒性。
能够评估特征的重要性。
此外，随机森林还具有一定的抗过拟合能力，因为通过随机选择特征和数据的过程，每个决策树都是在不同的子集上训练的，
可以减少单个决策树的过度拟合风险。

随机森林广泛应用于各种机器学习任务，包括分类、回归、特征选择等。它在实际应用中表现出良好的性能和稳定性，
并且相对于单个决策树来说，更能适应不同的数据情况和问题领域。
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0 ) #n_estimators 決策樹數量
"""
criterion参数用于指定决策树划分标准的选择方式。当criterion设为'entropy'时，表示使用信息熵（entropy）作为决策树的划分标准。

信息熵是衡量数据纯度和不确定性的一种指标。在决策树算法中，使用信息熵作为划分标准可以使得每次划分都最大程度地减少样本集合中
的不确定性和混乱度。在每个决策树的节点处，通过计算每个特征的信息增益，选择使得信息增益最大的特征作为最佳划分特征。

选择使用信息熵作为划分标准的随机森林模型可以在每个决策树中使用熵来评估特征的重要性，并根据最大化信息增益的原则进行划分。
这有助于提高模型的准确性和泛化能力。

需要注意的是，除了信息熵（entropy）外，还可以使用其他划分标准，如基尼系数（gini coefficient），
通过设置criterion参数为'gini'来选择基尼系数作为划分标准。不同的划分标准可能会对模型的性能产生影响，
具体选择哪种划分标准取决于数据集和问题的特点。
"""
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j, s=15)
plt.title('Random Forest (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j, s=15)
plt.title('Random Forest (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()