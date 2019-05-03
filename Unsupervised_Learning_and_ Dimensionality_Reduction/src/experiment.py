 # -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import homogeneity_completeness_v_measure, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

BASE_DIR = './../data/'
RANDOM_STATE = 0
DEFAULT_PERCENT = .7
CMAP = cm.get_cmap("Spectral")

#################################################
# get data

df1 = pd.read_csv(BASE_DIR + 'sky.csv')
df1 = df1.drop(['objid', 'rerun'], axis=1)
X1 = df1.iloc[:, :15].values
scale = MinMaxScaler()
scale.fit(X1)
X1 = scale.transform(X1)
y1 = df1.iloc[:, 15].values
le = LabelEncoder()
y1 = le.fit_transform(y1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=1-DEFAULT_PERCENT)


df2 = pd.read_csv(BASE_DIR + 'winequality.csv')
X2 = df2.iloc[:, :11].values
scale = MinMaxScaler()
scale.fit(X2)
normX2 = scale.transform(X2)
y2 = df2.iloc[:, 11].values
le = LabelEncoder()
y2 = le.fit_transform(y2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=1-DEFAULT_PERCENT)

#################################################
# KMeans

range_n_clusters = list(range(2, 11))
sse1 = []
sse2 = []
homo1 = []
homo2 = []
comp1 = []
comp2 = []
ac1 = []
ac2 = []

for n_clusters in range_n_clusters:
    clusterer1 = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit(X1)
    clusterer2 = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit(X2)
    cluster_labels1 = clusterer1.labels_
    cluster_labels2 = clusterer2.labels_
    sse1.append(clusterer1.inertia_)
    sse2.append(clusterer2.inertia_)
    ac1.append(accuracy_score(y1, cluster_labels1))
    ac2.append(accuracy_score(y2, cluster_labels2))
    homo1.append(homogeneity_completeness_v_measure(y1, cluster_labels1)[0])
    homo2.append(homogeneity_completeness_v_measure(y2, cluster_labels2)[1])
    comp1.append(homogeneity_completeness_v_measure(y1, cluster_labels1)[0])
    comp2.append(homogeneity_completeness_v_measure(y2, cluster_labels2)[1])

print('accuracy for sky survey data is: ', ac1)
print('accuracy for wine quality data is: ', ac2)
print('homogeneity for sky survey data is: ', homo1)
print('homogeneity for wine quality data is: ', homo2)
print('completeness for sky survey data is: ', comp1)
print('completeness for wine quality data is: ', comp2)

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, sse1, 'b*-', label='SSE')
ax.plot(3, sse1[1], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='g', markerfacecolor='None')
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.grid(True)
plt.legend(loc='best')
plt.title('Elbow curve for Sky Survey data')
fig.savefig('./../img/elbow1.png')
plt.show()

sse2 = list(map(lambda i: i / 1000, sse2))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, sse2, 'b*-', label='SSE')
ax.plot(4, sse2[2], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='g', markerfacecolor='None')
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares (*10^3)')
plt.grid(True)
plt.legend(loc='best')
plt.title('Elbow curve for Wine Quality data')
fig.savefig('./../img/elbow2.png')
plt.show()

#################################################
# plot the best cluster

# clusterer1 = KMeans(n_clusters=5, random_state=RANDOM_STATE).fit(X1)
# clusterer2 = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(X2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# colors = CMAP(clusterer1.labels_.astype(float) / 5)
# ax.scatter(pd.DataFrame(X1).iloc[:, 0], pd.DataFrame(X1).iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
# centers = clusterer1.cluster_centers_
# ax.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
# for i, c in enumerate(centers):
#     ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')

# ax.set_title("The visualization of the clustered data.")
# ax.set_xlabel("Feature space for the 1st feature")
# ax.set_ylabel("Feature space for the 2nd feature")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# colors = CMAP(clusterer1.labels_.astype(float) / 4)
# ax.scatter(pd.DataFrame(X2).iloc[:, 0], pd.DataFrame(X2).iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
# centers = clusterer2.cluster_centers_
# ax.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
# for i, c in enumerate(centers):
#     ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')

# ax.set_title("The visualization of the clustered data.")
# ax.set_xlabel("Feature space for the 1st feature")
# ax.set_ylabel("Feature space for the 2nd feature")
# plt.show()

#################################################
# EM

range_n_clusters = list(range(2, 11))
lh1 = []
lh2 = []
homo1 = []
homo2 = []
comp1 = []
comp2 = []
ac1 = []
ac2 = []

for n_clusters in range_n_clusters:
    gmm1 = GaussianMixture(n_components=n_clusters, random_state=10, covariance_type='tied').fit(X1)
    gmm2 = GaussianMixture(n_components=n_clusters, random_state=10, covariance_type='tied').fit(X2)
    y_pred1 = gmm1.predict(X1)
    y_pred2 = gmm2.predict(X2)
    lh1.append(gmm1.score(X1))
    lh2.append(gmm2.score(X2))
    ac1.append(accuracy_score(y1, y_pred1))
    ac2.append(accuracy_score(y2, y_pred2))
    homo1.append(homogeneity_completeness_v_measure(y1, y_pred1)[0])
    homo2.append(homogeneity_completeness_v_measure(y2, y_pred2)[0])
    comp1.append(homogeneity_completeness_v_measure(y1, y_pred1)[1])
    comp2.append(homogeneity_completeness_v_measure(y2, y_pred2)[1])

print('accuracy for sky survey data is: ', ac1)
print('accuracy for wine quality data is: ', ac2)
print('homogeneity for sky survey data is: ', homo1)
print('homogeneity for wine quality data is: ', homo2)
print('completeness for sky survey data is: ', comp1)
print('completeness for wine quality data is: ', comp2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, lh1, 'b*-', label='Log Likelihood')
plt.grid(True)
plt.xlabel('Number of mixture components')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve for Sky Survey data')
plt.legend(loc='best')
fig.savefig('./../img/log_likelihood1.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range_n_clusters, lh2, 'b*-', label='Log Likelihood')
plt.grid(True)
plt.xlabel('Number of mixture components')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve for Sky Survey data')
plt.legend(loc='best')
fig.savefig('./../img/log_likelihood2.png')
plt.show()

#################################################
# PCA

pca = PCA(n_components=5)
pca.fit(X1)
bins = np.linspace(-.001, .001, 100)
fig = plt.figure()
plt.title("Eigenvalue distribution for PCA with 5 components: Sky Survey data")
plt.xlabel('eigenvalue')
plt.ylabel('frequency')
for count, i in enumerate(pca.components_):
    plt.hist(i, bins, alpha=0.5, label=str(count+1))
plt.legend(loc='best')
fig.savefig('./../img/pca1.png')
plt.show()

pca = PCA(n_components=5)
pca.fit(X2)
bins = np.linspace(-.01, .01, 100)
fig = plt.figure()
plt.title("Eigenvalue distribution for PCA with 5 components: Wine Quality data")
plt.xlabel('eigenvalue')
plt.ylabel('frequency')
for count, i in enumerate(pca.components_):
    plt.hist(i, bins, alpha=0.5, label=str(count+1))
plt.legend(loc='best')
fig.savefig('./../img/pca2.png')
plt.show()

train_acc = []
test_acc = []
n = range(2, 15, 1)
model = KNeighborsClassifier(10, weights='distance')

for i in n:
    pca = PCA(n_components=i)
    pca.fit(X1_train)
    X_new = pca.transform(X1_train)
    X_test_new = pca.transform(X1_test)
    model.fit(X_new, y1_train)
    train_acc.append(accuracy_score(y1_train, model.predict(X_new)))
    test_acc.append(accuracy_score(y1_test, model.predict(X_test_new)))

fig = plt.figure()
plt.title('Accuracy Score for KNN for Sky Survey data with PCA')
plt.ylabel('accuracy')
plt.xlabel('number of components')
plt.plot(n, train_acc, 'b*-', label='train accuracy')
plt.plot(n, test_acc, 'r^-', label='test accuracy')
plt.legend(loc='best')
plt.grid(True)
fig.savefig('./../img/pca3.png')
plt.show()

train_acc = []
test_acc = []
n = range(2, 11, 1)
model = KNeighborsClassifier(10, weights='distance')

for i in n:
    pca = PCA(n_components=i)
    pca.fit(X2_train)
    X_new = pca.transform(X2_train)
    X_test_new = pca.transform(X2_test)
    model.fit(X_new, y2_train)
    train_acc.append(accuracy_score(y2_train, model.predict(X_new)))
    test_acc.append(accuracy_score(y2_test, model.predict(X_test_new)))

fig = plt.figure()
plt.title('Accuracy Score for KNN for Wine Quality data with PCA')
plt.ylabel('accuracy')
plt.xlabel('number of components')
plt.plot(n, train_acc, 'b*-', label='train accuracy')
plt.plot(n, test_acc, 'r^-', label='test accuracy')
plt.legend(loc='best')
plt.grid(True)
fig.savefig('./../img/pca4.png')
plt.show()

#################################################
# ica 

ica = FastICA(n_components=5)
ica.fit(X1)
bins = np.linspace(-.0001, .0001, 100)
fig = plt.figure()
plt.title("Components distribution for ICA with 5 components: Sky Survey data")
plt.xlabel('value')
plt.ylabel('frequency')
a = []
for count, i in enumerate(ica.components_):
    a.extend(i)
    kurt = stats.kurtosis(i)
    plt.hist(i, bins, alpha=0.5, label=str(count+1)+": "+str(kurt))
        
plt.legend(loc='best')
print(stats.kurtosis(a))
fig.savefig('./../img/ica1.png')
plt.show()

ica = FastICA(n_components=5)
ica.fit(X2)
bins = np.linspace(-.001, .001, 100)
fig = plt.figure()
plt.title("Components distribution for ICA with 5 components: Wine Quality data")
plt.xlabel('value')
plt.ylabel('frequency')
a = []
for count, i in enumerate(ica.components_):
    a.extend(i)
    kurt = stats.kurtosis(i)
    plt.hist(i, bins, alpha=0.5, label=str(count+1)+": "+str(kurt))
        
plt.legend(loc='best')
print(stats.kurtosis(a))
fig.savefig('./../img/ica2.png')
plt.show()

train_acc = []
test_acc = []
n = range(2, 15, 1)
model = KNeighborsClassifier(10, weights='distance')

for i in n:
    ica = FastICA(n_components=i, random_state=RANDOM_STATE)
    ica.fit(X1_train)
    X_new = ica.transform(X1_train)
    X_test_new = ica.transform(X1_test)
    model.fit(X_new, y1_train)
    train_acc.append(accuracy_score(y1_train, model.predict(X_new)))
    test_acc.append(accuracy_score(y1_test, model.predict(X_test_new)))

fig = plt.figure()
plt.title('Accuracy Score for KNN for Sky Survey data with ICA')
plt.ylabel('accuracy')
plt.xlabel('number of components')
plt.plot(n, train_acc, 'b*-', label='train accuracy')
plt.plot(n, test_acc, 'r^-', label='test accuracy')
plt.legend(loc='best')
plt.grid(True)
fig.savefig('./../img/ica3.png')
plt.show()

train_acc = []
test_acc = []
n = range(2, 11, 1)
model = KNeighborsClassifier(10, weights='distance')

for i in n:
    ica = FastICA(n_components=i, random_state=RANDOM_STATE)
    ica.fit(X2_train)
    X_new = ica.transform(X2_train)
    X_test_new = ica.transform(X2_test)
    model.fit(X_new, y2_train)
    train_acc.append(accuracy_score(y2_train, model.predict(X_new)))
    test_acc.append(accuracy_score(y2_test, model.predict(X_test_new)))

fig = plt.figure()
plt.title('Accuracy Score for KNN for Wine Quality data with ICA')
plt.ylabel('accuracy')
plt.xlabel('number of components')
plt.plot(n, train_acc, 'b*-', label='train accuracy')
plt.plot(n, test_acc, 'r^-', label='test accuracy')
plt.legend(loc='best')
plt.grid(True)
fig.savefig('./../img/ica4.png')
plt.show()

#################################################
# RP

rp = GaussianRandomProjection(n_components=10, random_state=RANDOM_STATE)
rp.fit(X1)
bins = np.linspace(-1, 1, 100)
fig = plt.figure()
plt.title("Components distribution for RP with 4 components: Sky Survey data")
plt.xlabel('value')
plt.ylabel('frequency')
a = []
for count, i in enumerate(rp.components_):
    a.extend(i)
    plt.hist(i, bins, alpha=0.5, label=str(count+1))
plt.legend(loc='best')
print(stats.kurtosis(a))
fig.savefig('./../img/rp3.png')
plt.show()

train_acc = []
test_acc = []
train_acc_avg = []
test_acc_avg = []
n = range(2, 15, 1)
model = KNeighborsClassifier(10, weights='distance')

for i in n:
    for j in range(20):
        ica = GaussianRandomProjection(n_components=i, random_state=RANDOM_STATE)
        ica.fit(X1_train)
        X_new = ica.transform(X1_train)
        X_test_new = ica.transform(X1_test)
        model.fit(X_new, y1_train)
        train_acc.append(accuracy_score(y1_train, model.predict(X_new)))
        test_acc.append(accuracy_score(y1_test, model.predict(X_test_new)))
    train_acc_avg.append(np.mean(train_acc))
    test_acc_avg.append(np.mean(test_acc))

fig = plt.figure()
plt.title('Accuracy Score for KNN for Sky Survey data with RP in 20 iterations')
plt.ylabel('average accuracy')
plt.xlabel('number of components')
plt.plot(n, train_acc_avg, 'b*-', label='train accuracy')
plt.plot(n, test_acc_avg, 'r^-', label='test accuracy')
plt.legend(loc='best')
plt.grid(True)
fig.savefig('./../img/rp4.png')
plt.show()

train_acc = []
test_acc = []
train_acc_avg = []
test_acc_avg = []
n = range(2, 11, 1)
model = KNeighborsClassifier(10, weights='distance')

for i in n:
    for j in range(20):
        ica = GaussianRandomProjection(n_components=i, random_state=RANDOM_STATE)
        ica.fit(X2_train)
        X_new = ica.transform(X2_train)
        X_test_new = ica.transform(X2_test)
        model.fit(X_new, y2_train)
        train_acc.append(accuracy_score(y2_train, model.predict(X_new)))
        test_acc.append(accuracy_score(y2_test, model.predict(X_test_new)))
    train_acc_avg.append(np.mean(train_acc))
    test_acc_avg.append(np.mean(test_acc))

fig = plt.figure()
plt.title('Accuracy Score for KNN for Wine Quality data with RP in 20 iterations')
plt.ylabel('average accuracy')
plt.xlabel('number of components')
plt.plot(n, train_acc_avg, 'b*-', label='train accuracy')
plt.plot(n, test_acc_avg, 'r^-', label='test accuracy')
plt.legend(loc='best')
plt.grid(True)
fig.savefig('./../img/rp5.png')
plt.show()

#################################################
# cluster with DR
def dim_reduct(name, X_train, y_train, X_test, y_test, title, n, cluster):
    if name == 'em':
        model = KMeans(n_clusters=cluster, random_state=RANDOM_STATE)
    if name == 'kmeans':
        model = GaussianMixture(n_components=cluster, random_state=RANDOM_STATE)
    
    comp = range(2, n, 1)
    
    methods = [PCA, FastICA, NMF, GaussianRandomProjection]
    names = ['PCA', 'ICA', 'NMF', 'RP']
    res = []
    
    for meth_c, dr in enumerate(methods):
        r_test = []
        for j in comp:
            x_test = []
            
            if names[meth_c] == 'RP':
                its = 20
            else:
                its = 1
            
            for it in range(its):
                dr_i = dr(n_components=j)
                X_new = dr_i.fit_transform(X_train)
                X_new_test = dr_i.fit_transform(X_test)
            
                model.fit(X_new)
                    
                acc_test = accuracy_score(y_test, model.predict(X_new_test))
                
                x_test.append(acc_test)
                
                del dr_i
                del X_new
                del X_new_test

            r_test.append(np.mean(x_test))
        res.append(r_test)

               
    fig = plt.figure()
    
    plt.plot(comp, res[0], 'b*-', label='PCA')
    plt.plot(comp, res[1], 'g^-', label='ICA')
    plt.plot(comp, res[2], 'rs-', label='NMF')
    plt.plot(comp, res[3], 'y+-', label='RP')
    plt.title('Comparing dimensionality reduction with ' + name + ' for ' + title)
    plt.xlabel('N Components')
    plt.ylabel('Classification accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    fig.savefig('./../img/dim_rect_' + name + '_' + str(cluster) + '.png')
    plt.show()

dim_reduct('em', X1_train, y1_train, X1_test, y1_test, 'Sky survey data', 15, 3)
dim_reduct('em', X2_train, y2_train, X2_test, y2_test, 'Wine Quality data', 11, 7)
dim_reduct('kmeans', X1_train, y1_train, X1_test, y1_test, 'Sky survey data', 15, 3)
dim_reduct('kmeans', X2_train, y2_train, X2_test, y2_test, 'Wine Quality data', 11, 7)

#################################################
# NN with DR

def nn_dim_reduct():
    model = MLPClassifier(hidden_layer_sizes=(5,5))      
    
    comp = [2,4,6,8,10,12,14]    
    
    methods = [PCA, FastICA, GaussianRandomProjection]
    names = ['PCA', 'ICA', 'RP']
    res = []
    res_test = []
    
    for meth_c, dr in enumerate(methods):
        print(names[meth_c])
        r = []
        r_test = []

        for j in comp:
            x = []
            x_test = []
            
            
            for it in range(20):
                dr_i = dr(n_components=j)
                X_new = dr_i.fit_transform(X1_train)
                X_new_test = dr_i.transform(X1_test)
            
                model.fit(X_new, y1_train)
                    
                acc = accuracy_score(y1_train, model.predict(X_new))
                acc_test = accuracy_score(y1_test, model.predict(X_new_test))
                
                x.append(acc)
                x_test.append(acc_test)
                
                del dr_i
                del X_new
                del X_new_test

            r.append(np.mean(x))
            r_test.append(np.mean(x_test))
        res.append(r)
        res_test.append(r_test)

               
    fig = plt.figure()
    COLORS = 'bygr'

    for count, i in enumerate(methods):
        plt.plot(comp, res[count], label=names[count]+' training', color=COLORS[count])
        plt.plot(comp, res_test[count], label=names[count]+' testing', color=COLORS[count])

    plt.title("NN Comparing dimensionality reduction techniques")
    plt.xlabel('N Components')
    plt.ylabel('Classification accuracy')

    plt.legend(loc='best')
    fig.savefig('./../img/dim_rect_nn.png')
    plt.show()

nn_dim_reduct()

#################################################
# NN with cluster with DR

def nn_dim_reduct_cluster(name, k=3):
    if name == 'em':
        clust = GaussianMixture(n_components=k, covariance_type='spherical')
    elif name == 'kmeans':
        clust = KMeans(n_clusters=k, random_state=0)
    else:
        raise Exception('enter kmeans or em as name')
    
    model = MLPClassifier(hidden_layer_sizes=(5,5))      
    
    comp = [2,4,6,8,10,12,14]    
    
    methods = [FastICA, PCA, GaussianRandomProjection]
    names = ['ICA', 'PCA', 'RP']
    res = []
    res_test = []
    
    for meth_c, dr in enumerate(methods):
        print(names[meth_c])
        r = []
        r_test = []

        for j in comp:
            x = []
            x_test = []
            
            print(j)
            
            for it in range(20):
                dr_i = dr(n_components=j)
                X_new = dr_i.fit_transform(X1_train)
                X_new_test = dr_i.transform(X1_test)
                
                clust.fit(X_new)
                clusters = clust.predict(X_new)
                clusters_test = clust.predict(X_new_test)
                
                clusters = clusters.reshape(clusters.shape[0], 1)
                clusters_test = clusters_test.reshape(clusters_test.shape[0], 1)

                data = np.hstack([X_new, clusters])
                data_test = np.hstack([X_new_test, clusters_test])
            
                model.fit(data, y1_train)
                    
                acc = accuracy_score(y1_train, model.predict(data))
                acc_test = accuracy_score(y1_test, model.predict(data_test))
                
                x.append(acc)
                x_test.append(acc_test)
                
                del dr_i
                del X_new
                del X_new_test

            r.append(np.mean(x))
            r_test.append(np.mean(x_test))
        res.append(r)
        res_test.append(r_test)

               
    fig = plt.figure()
    COLORS = 'bygr'

    for count, i in enumerate(methods):
        plt.plot(comp, res[count], label=names[count]+' training', color=COLORS[count])
        plt.plot(comp, res_test[count], label=names[count]+' testing', color=COLORS[count])
    if k == 3:
        plt.title("NN with custers: Comparing dimensionality reduction techniques")
    else:
        plt.title("NN with custers: Comparing DR techniques, k=" + str(k))
    plt.xlabel('N Components')
    plt.ylabel('Classification accuracy')

    plt.legend(loc='best')
    if k == 3:
        fig.savefig('./../img/dim_rect_nn_cluster_' + name + '.png')
    else:
        fig.savefig('./../img/dim_rect_nn_cluster_' + str(k) + '.png')
    plt.show()

nn_dim_reduct_cluster('kmeans')
nn_dim_reduct_cluster('em')

nn_dim_reduct_cluster('kmeans', 5)
nn_dim_reduct_cluster('kmeans', 10)