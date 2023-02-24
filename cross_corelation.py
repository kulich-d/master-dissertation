import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tslearn.metrics.dtw_variants import dtw_path

## для двойной опоры которая в начале шага посмтреть общие экстремумы леовй ноги: максимального угла, x минимальной пятки и x минимального носка
# y_3 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")[:, 0]
# y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")[:, 0]
# y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_feet_angle.txt")
# y_3 = 1 - y_3
# y_2 = 1 - y_2


y_3 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_foot_index.txt")[:, 0]
y_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_heel.txt")[:, 0]
y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_feet_angle.txt")
y_3 = 1 - y_3
y_2 = 1 - y_2

print(np.correlate(y, y_2, "full"))
cor = np.correlate(y, y_2, "full")
print(np.argmax(cor))
x = list(range(len(y)))
path_dtw = dtw_path(y, y_2)[0]
path_dtw_1 = {a[0]: a[1] for a in path_dtw}
path_dtw_2 = {a[1]: a[0] for a in path_dtw}

peaks_y = find_peaks(y, height=0, distance=20)[0]
print(peaks_y)
peaks_y_2 = find_peaks(y_2, height=0, distance=20)[0]
peaks_y_3 = find_peaks(y_3, height=0, distance=20)[0]
print(peaks_y_2)
print(peaks_y_3)
all_picks = []
all_picks_mean = []
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
plt.plot(x, y, '-', label='right feet angle')
plt.plot(x, y_2, '-', label='right foot index')
plt.legend()
plt.show()
