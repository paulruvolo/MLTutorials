import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

data = load_boston()
train_percentages = range(5,95,5)
test_r2 = numpy.zeros(len(train_percentages))

for (i,train_percent) in enumerate(train_percentages):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_percent/100.0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_r2[i] = model.score(X_test, y_test)

fig = plt.figure()
plt.plot(train_percentages, test_r2)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('R^2')
plt.show()
