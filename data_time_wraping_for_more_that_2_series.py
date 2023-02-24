import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tslearn.metrics.dtw_variants import dtw_path
from scipy.signal import savgol_filter

## для двойной опоры посмтреть общие экстремумы леовй ноги: максимального угла, x минимальной пятки и x минимального носка
# y_3 = np.loadtxt("dtw_barycenter_averaging_subgradient.txt")
# y_3 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 0]
# y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
# y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
# y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
y = np.loadtxt("dtw_barycenter_averaging_subgradient_angle.txt")# y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
y_2 = np.loadtxt("dtw_barycenter_averaging_subgradient_leg.txt")# y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
# y_3 = 1 - y_3
# y_2 = 1 - y_2
#
# all_info = [y, y_2, y_3]
#
# all_dtw_1 = []
# all_dtw_2 = []
# for y in all_info[-1]:
#     for y_2 in all_info[1:]:
#         path_dtw = dtw_path(y, y_2)[0]
#         path_dtw_1 = {a[0]: a[1] for a in path_dtw}
#         path_dtw_2 = {a[1]: a[0] for a in path_dtw}
#         all_dtw_1.append(path_dtw_1)
#         all_dtw_2.append(path_dtw_2)
#
# all_peaks = []
# for y in all_info:
#     all_peaks.append(find_peaks(y, height=0, distance=20)[0])
#
# all_picks = []
# all_picks_mean = []
# for i, p in enumerate(all_peaks[-1]):
#     for j, p_1 in enumerate(all_peaks[1:]):
#         if all_dtw_1[i][p] in all_peaks[j]:
#             all_picks.append(p)
#         if all_dtw_2[j][p_1] in all_peaks[i]:
#             all_picks.append(p)
# print(all_picks)
#

all_picks = []
all_picks_mean = []
x = list(range(len(y)))
path_dtw = dtw_path(y, y_2)[0]
path_dtw_1 = {a[0]: a[1] for a in path_dtw}
path_dtw_2 = {a[1]: a[0] for a in path_dtw}

peaks_y = find_peaks(y, height=0, distance=20)[0]
print(peaks_y)
peaks_y_2 = find_peaks(y_2, height=0, distance=20)[0]
print(peaks_y_2)

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

plt.plot(peaks_y, y[peaks_y], "x")
plt.plot(peaks_y_2, y_2[peaks_y_2], "x")

# plt.plot(list(range(len(cor))), cor, label="cor")
plt.plot(x, y, '-', label='foot index')
plt.plot(x, y_2, '-', label='knee')
plt.show()

# from tslearn.barycenters import dtw_barycenter_averaging
# b = dtw_barycenter_averaging(dataset)
# print(b)
