import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt

# y_o = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_heel.txt")
# y_o_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_heel.txt")
# x_o = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")


# rule for  1-rst одиночная опора
# y_o = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_len_feet.txt")
# y_o_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")
#
# y_o = y_o
# y_o_2 = 1 - y_o_2[:, 0]
#
# peaks, _ = find_peaks(y_o, height=0, distance=20)
# print(peaks)
#
# peaks_2, _ = find_peaks(y_o_2, height=0, distance=20)
# print(peaks_2)
#
# plt.plot(y_o)
# plt.plot(y_o_2)
#
# plt.plot(peaks, y_o[peaks], "x")
# plt.plot(peaks_2, y_o_2[peaks_2], "x")
#
# plt.show()

# rule for  2-rst одиночная опора
# y_o = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_len_feet.txt")
# y_o_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_foot_index.txt")
#
#
# y_o = y_o
# y_o_2 = 1 - y_o_2[:, 0]
#
# peaks, _ = find_peaks(y_o, height=0)
# print(peaks)
#
# peaks_2, _ = find_peaks(y_o_2, height=0)
# print(peaks_2)
#
# plt.plot(y_o)
# plt.plot(y_o_2)
#
# plt.plot(peaks, y_o[peaks], "x")
# plt.plot(peaks_2, y_o_2[peaks_2], "x")
#
# plt.show()


# rule for  двойная опора
# y_o = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_len_feet.txt")
# y_o_2 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_right_len_feet.txt")
# y_o_3 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")
y_o_3 = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_knee.txt")

# y_o = y_o
# y_o_2 = 1 - y_o_2
y_o_3 = y_o_3[:,1]

# summup = np.array([a+b for a, b in zip(y_o, y_o_2)])

# peaks, _ = find_peaks(y_o, height=0,)
# peaks_1, _ = find_peaks(y_o_2, height=0)
y_o_3_gradient = 1 - np.gradient(y_o_3)
peaks_2, _ = find_peaks(y_o_3, height=0)
peaks_2_grad, _ = find_peaks(y_o_3_gradient, height=0,distance=20)
# print(_peaks)
# print(peaks_1)
print(peaks_2)
print(peaks_2_grad)

# plt.plot(y_o)
# plt.plot(y_o_2)
plt.plot(y_o_3)
plt.plot(y_o_3_gradient)

# plt.plot(peaks, y_o[peaks], "x")
# plt.plot(peaks_1, y_o_2[peaks_1], "x")
plt.plot(peaks_2, y_o_3[peaks_2], "x")
plt.plot(peaks_2_grad, y_o_3_gradient[peaks_2_grad], "x")


#
# peaks_2= find_peaks_cwt(y_o_3, widths=10)
# peaks_2_grad = find_peaks_cwt(y_o_3_gradient, widths=10)
# # print(_peaks)
# # print(peaks_1)
# print(peaks_2)
# print(peaks_2_grad)
#
# # plt.plot(y_o)
# # plt.plot(y_o_2)
# plt.plot(y_o_3)
# plt.plot(y_o_3_gradient)
#
# # plt.plot(peaks, y_o[peaks], "x")
# # plt.plot(peaks_1, y_o_2[peaks_1], "x")
# plt.plot(peaks_2, y_o_3[peaks_2], "x")
# plt.plot(peaks_2_grad, y_o_3_gradient[peaks_2_grad], "x")

plt.show()

plt.show()
