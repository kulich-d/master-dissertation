# Author: Romain Tavenard, Felix Divo
# License: BSD 3 clause

import os

import matplotlib.pyplot as plt
import numpy as np
from tslearn.barycenters import \
    dtw_barycenter_averaging_subgradient

save_path = "/Users/diana.kulich/Documents/Masters/dissertation/exp/check_new supportch01_20180308130614"
side = "left"
x_heel = 1 - np.loadtxt(os.path.join(save_path, f"filter_{side}_heel.txt"))[0, :]
x_foot_index = 1 - np.loadtxt(os.path.join(save_path, f"filter_{side}_foot_index.txt"))[0, :]
X = np.stack([x_heel, x_foot_index])
coordinates_barycenter = dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3)
angle_left = np.loadtxt(os.path.join(save_path, "filter_left_feet_angle.txt"))
angle_right = 1 - np.loadtxt(os.path.join(save_path, "filter_left_knee.txt"))[1, :]
X = np.stack([angle_left, angle_right])
angle_barycenter = dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-3)
# y_3 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
# y_2 = numpy.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_feet_angle.txt")
# y_2 = 1 - y_2

# fetch the example data set
# numpy.random.seed(0)
# # X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
# X = numpy.stack([y_3, y_2, y_1])
# # X = numpy.stack([y_3, y_2])
# length_of_sequence = X.shape[1]
#
#
# def plot_helper(barycenter):
#     # plot all points of the data set
#     for series in X:
#         plt.plot(series.ravel(), "k-", alpha=.2)
#     # plot the given barycenter of them
#     plt.plot(barycenter.ravel(), "r-", linewidth=2)


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
# dtw_barycenter = dtw_barycenter_averaging(X, max_iter=50, tol=1e-3)
# dtw_barycenter = dtw_barycenter_averaging_subgradient(X, max_iter=50, tol=1e-5)
# plt.plot(dtw_barycenter)
# for series in X:
#     plt.plot(series.ravel(), "k-", alpha=.2)
# plt.title("DBA (subgradient descent approach)")
# plt.show()

# numpy.savetxt("dtw_barycenter_averaging_subgradient_leg.txt", dtw_barycenter)
from scipy.signal import find_peaks

coordinates_barycenter = np.squeeze(coordinates_barycenter)
angle_barycenter = np.squeeze(angle_barycenter)
peaks_y = find_peaks(coordinates_barycenter, height=0, distance=20)[0]
print(peaks_y)
peaks_y_2 = find_peaks(angle_barycenter, height=0, distance=20)[0]
print(peaks_y_2)

all_picks_mean = []
all_picks = []
from tslearn.metrics.dtw_variants import dtw_path

path_dtw = dtw_path(coordinates_barycenter, angle_barycenter)[0]
path_dtw_1 = {a[0]: a[1] for a in path_dtw}
path_dtw_2 = {a[1]: a[0] for a in path_dtw}

for p in peaks_y:
    if path_dtw_1[p] in peaks_y_2:
        all_picks.append(p)
        all_picks_mean.append(np.mean([p, path_dtw_1[p]]))
for p in peaks_y_2:
    if path_dtw_2[p] in peaks_y:
        all_picks.append(p)
        all_picks_mean.append(np.mean([p, path_dtw_2[p]]))

print(f"common_picks: {all_picks}")
print(f"common_mean_picks: {all_picks_mean}")
plt.plot(coordinates_barycenter)
plt.plot(angle_barycenter)
plt.title("DBA (subgradient descent approach)")
plt.show()
