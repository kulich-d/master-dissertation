# Author: Romain Tavenard, Felix Divo
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt
from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, \
    softdtw_barycenter
from tslearn.datasets import CachedDatasets

y_3 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 0]
y_2 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
y_1 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_knee.txt")[:, 0]
y = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
y_3 = 1 - y_3
y_2 = 1 - y_2
y_1 = 1 - y_1

# y_3 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
# y_2 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_feet_angle.txt")
# y_2 = 1 - y_2

# fetch the example data set
numpy.random.seed(0)
# X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
X = numpy.stack([y_3, y_2, y_1])
# X = numpy.stack([y_3, y_2])
length_of_sequence = X.shape[1]


def plot_helper(barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)


# plot the four variants with the same number of iterations and a tolerance of
# 1e-3 where applicable
# ax1 = plt.subplot(4, 1, 1)
# plt.title("Euclidean barycenter")
# plot_helper(euclidean_barycenter(X))
#
# plt.subplot(4, 1, 2, sharex=ax1)
# plt.title("DBA (vectorized version of Petitjean's EM)")
# plot_helper(dtw_barycenter_averaging(X, max_iter=50, tol=1e-3))
#
# plt.subplot(4, 1, 3, sharex=ax1)
# plt.title("DBA (subgradient descent approach)")
# plot_helper(dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3))
#
# plt.subplot(4, 1, 4, sharex=ax1)
# plt.title("Soft-DTW barycenter ($\gamma$=1.0)")
# plot_helper(softdtw_barycenter(X, gamma=1., max_iter=50, tol=1e-3))
#
# # clip the axes for better readability
# ax1.set_xlim([0, length_of_sequence])
#
# # show the plot(s)
# plt.tight_layout()
dtw_barycenter = dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3)
plt.plot(dtw_barycenter)
for series in X:
    plt.plot(series.ravel(), "k-", alpha=.2)
plt.title("DBA (subgradient descent approach)")
plt.show()

numpy.savetxt("dtw_barycenter_averaging_subgradient_leg.txt", dtw_barycenter)
