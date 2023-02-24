import matplotlib.pyplot as plt
import numpy as np

y = np.loadtxt("/Users/diana.kulich/Documents/Masters/dissertation/personal_idea/collect_left_foot_index.txt")
x = np.array(list(range(len(y))))

# plt.figure()
plt.plot(x, y[:, 1])
# plt.xlim(0, 22)
# plt.xlabel('Period')
# plt.ylabel('Power ($\cdot10^3$)')

plt.show()