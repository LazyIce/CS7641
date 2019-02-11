import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


RANDOM_STATE = 42

data = pd.read_csv("../data/sky.csv")
print(data.info())
print(data.describe())
print(list(data.isnull().any()))

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1:].values

labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(np.ravel(Y))


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=0.25)
mm = MinMaxScaler()
X_train_scaled = mm.fit_transform(X_train)
X_test_scaled = mm.fit_transform(X_test)

# Decision Tree classifier
# decision tree learning curve of different sizes of test set
def getTestSizeCurve(max_depth, index):
    list1 = []
    list2 = []
    for i in range(10, 91):
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=RANDOM_STATE)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=1 - i / 100)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))
    plt.figure()
    plt.plot(range(10, 91), list1, label="train")
    plt.plot(range(10, 91), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# decision tree learning curve of different max_depth with gini and entropy
def getMaxDepthWithGiniandEntropy(index):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for i in range(2, 31):
        clf1 = DecisionTreeClassifier(max_depth=i, criterion="gini", random_state=RANDOM_STATE)

        clf1.fit(X_train_scaled, Y_train)
        train_predict = clf1.predict(X_train_scaled)
        test_predict = clf1.predict(X_test_scaled)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

        clf2 = DecisionTreeClassifier(max_depth=i, criterion="entropy", random_state=RANDOM_STATE)
        clf2.fit(X_train_scaled, Y_train)
        train_predict = clf2.predict(X_train_scaled)
        test_predict = clf2.predict(X_test_scaled)
        list3.append(accuracy_score(Y_train, train_predict))
        list4.append(accuracy_score(Y_test, test_predict))
    plt.figure()
    plt.plot(range(2, 31), list1, label="train with gini")
    plt.plot(range(2, 31), list2, label="test with gini")
    plt.plot(range(2, 31), list3, label="train with entropy")
    plt.plot(range(2, 31), list4, label="test with entropy")
    plt.ylabel("accuracy")
    plt.xlabel("maximum depth")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Neural network classifier
# Neural network classifier learning curve of hidden_layer_sizes
def getHiddenLayerSizes(layer_size, index):
    list1 = []
    list2 = []
    for i in range(10, 91):
        clf = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=1000, random_state=RANDOM_STATE)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - i / 100, random_state=RANDOM_STATE)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="train")
    plt.plot(range(10, 91), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Neural network classifier learning curve of activations
def getActivations(index):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for i in range(10, 91):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - i / 100, random_state=RANDOM_STATE)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)

        clf1 = MLPClassifier(hidden_layer_sizes=5, activation="identity", max_iter=1000, random_state=RANDOM_STATE)
        clf1.fit(X_train, Y_train)
        test_predict = clf1.predict(X_test)
        list1.append(accuracy_score(Y_test, test_predict))

        clf2 = MLPClassifier(hidden_layer_sizes=5, activation="logistic", max_iter=1000, random_state=RANDOM_STATE)
        clf2.fit(X_train, Y_train)
        test_predict = clf2.predict(X_test)
        list2.append(accuracy_score(Y_test, test_predict))

        clf3 = MLPClassifier(hidden_layer_sizes=5, activation="tanh", max_iter=1000, random_state=RANDOM_STATE)
        clf3.fit(X_train, Y_train)
        test_predict = clf3.predict(X_test)
        list3.append(accuracy_score(Y_test, test_predict))

        clf4 = MLPClassifier(hidden_layer_sizes=5, activation="relu", max_iter=1000, random_state=RANDOM_STATE)
        clf4.fit(X_train, Y_train)
        test_predict = clf4.predict(X_test)
        list4.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="identity")
    plt.plot(range(10, 91), list2, label="logistic")
    plt.plot(range(10, 91), list3, label="tanh")
    plt.plot(range(10, 91), list4, label="relu")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Neural network classifier learning curve of solvers
def getSolvers(index):
    list1 = []
    list2 = []
    list3 = []
    for i in range(10, 91):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - i / 100, random_state=RANDOM_STATE)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)

        clf1 = MLPClassifier(hidden_layer_sizes=5, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
        clf1.fit(X_train, Y_train)
        test_predict = clf1.predict(X_test)
        list1.append(accuracy_score(Y_test, test_predict))

        clf2 = MLPClassifier(hidden_layer_sizes=5, solver="sgd", max_iter=1000, random_state=RANDOM_STATE)
        clf2.fit(X_train, Y_train)
        test_predict = clf2.predict(X_test)
        list2.append(accuracy_score(Y_test, test_predict))

        clf3 = MLPClassifier(hidden_layer_sizes=5, solver="adam", max_iter=1000, random_state=RANDOM_STATE)
        clf3.fit(X_train, Y_train)
        test_predict = clf3.predict(X_test)
        list3.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="lbfgs")
    plt.plot(range(10, 91), list2, label="sgd")
    plt.plot(range(10, 91), list3, label="adam")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Boosting classifier
# Boosting classifier learning curve of adaBoosting
def getAdaBoosting(index):
    list1 = []
    list2 = []
    for i in range(10, 91):
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=100, random_state=RANDOM_STATE)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=1 - i / 100)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="train")
    plt.plot(range(10, 91), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Boosting classifier learning curve of gradientBoosting
def getGradientBoosting(index):
    list1 = []
    list2 = []
    for i in range(10, 91):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=7, random_state=RANDOM_STATE)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=1 - i / 100)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="train")
    plt.plot(range(10, 91), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Boosting classifier learning curve of n_estimators
def getBoostingEstimators(index):
    list1 = []
    list2 = []
    for i in range(10, 210, 10):
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=i, random_state=RANDOM_STATE)
        clf.fit(X_train_scaled, Y_train)
        train_predict = clf.predict(X_train_scaled)
        test_predict = clf.predict(X_test_scaled)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 210, 10), list1, label="train")
    plt.plot(range(10, 210, 10), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("number of estimators")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# Boosting classifier learning curve of max depth of decision tree
def getBoostingMaxDepth(index):
    list1 = []
    list2 = []
    for i in range(2, 16):
        clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i), n_estimators=200, random_state=RANDOM_STATE)
        clf.fit(X_train_scaled, Y_train)
        train_predict = clf.predict(X_train_scaled)
        test_predict = clf.predict(X_test_scaled)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(2, 16), list1, label="train")
    plt.plot(range(2, 16), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("max depth of decision tree")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# svm classifier
# svm classifier learning curve
def getSVC(kernel, index):
    list1 = []
    list2 = []
    for i in range(10, 91):
        clf = svm.SVC(C=1, kernel=kernel, random_state=RANDOM_STATE)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=1 - i / 100)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="train")
    plt.plot(range(10, 91), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


# KNN classifier
# KNN classifier learning curve of best K
def getKNNBestK(index):
    list1 = []
    list2 = []
    for i in range(1, 31):
        clf = KNeighborsClassifier(n_neighbors=i, weights="distance")
        clf.fit(X_train_scaled, Y_train)
        train_predict = clf.predict(X_train_scaled)
        test_predict = clf.predict(X_test_scaled)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(1, 31), list1, label="train")
    plt.plot(range(1, 31), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("K neighbors")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")

# KNN classifier learning curve
def getKNN(index):
    list1 = []
    list2 = []
    for i in range(10, 91):
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=1 - i / 100)
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.fit_transform(X_test)
        clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        list1.append(accuracy_score(Y_train, train_predict))
        list2.append(accuracy_score(Y_test, test_predict))

    plt.figure()
    plt.plot(range(10, 91), list1, label="train")
    plt.plot(range(10, 91), list2, label="test")
    plt.ylabel("accuracy")
    plt.xlabel("size of test set(%)")
    plt.grid()
    plt.legend()
    plt.savefig("../img/figure" + str(index) + ".png")


def main():
    getTestSizeCurve(5, 20)
    getTestSizeCurve(8, 21)
    getMaxDepthWithGiniandEntropy(22)
    getHiddenLayerSizes(5, 23)
    getHiddenLayerSizes(20, 24)
    getHiddenLayerSizes((5, 20), 25)
    getHiddenLayerSizes((5, 10, 20), 26)
    getActivations(27)
    getSolvers(28)
    getAdaBoosting(29)
    getGradientBoosting(30)
    getBoostingEstimators(31)
    getBoostingMaxDepth(32)
    getSVC("linear", 33)
    getSVC("poly", 34)
    getSVC("rbf", 35)
    getSVC("sigmoid", 36)
    getKNNBestK(37)
    getKNN(38)
    return

if __name__ == "__main__":
    main()
