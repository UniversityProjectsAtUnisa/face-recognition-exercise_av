import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_bidimensional_data(X, y, colors, markers, x_label, y_label):
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0], X[y==l, 1], c = c, label = l, marker = m)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s','x','o','^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # Plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())
    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)
    plt.legend(loc='best')
    plt.show()

def plot_bar_chart(data, interval, x_label, y_label):
    plt.bar(interval, data, alpha = 0.5, align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.gca().set_ylim([min(acc)-0.1,max(acc)+0.1])
    plt.legend(loc='best')
    plt.show()

def plot_variance(variance, cum_variance, interval, x_label, y_label):
    plt.bar(interval, variance, alpha = 0.5, align='center', label = 'Individual variance')
    plt.step(interval, cum_variance, where='mid', label = 'Cumulative variance')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.show()