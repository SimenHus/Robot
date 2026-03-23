import matplotlib.pyplot as plt
import numpy as np

with open('./build/kalmanOutput.csv', 'r') as f:
    data = []
    for i, line in enumerate(f.readlines()):
        items = line.split(',')
        data.append([float(items[0]), float(items[1])])

with open('./build/groundTruth.csv', 'r') as f:
    gt = []
    for i, line in enumerate(f.readlines()):
        items = line.split(',')
        gt.append([float(items[0]), float(items[1])])

time = [i for i in range(len(data))]
data = np.array(data)
gt = np.array(gt)
plt.plot(data[:, 0], data[:, 1], label = 'Filter')
plt.plot(gt[:, 0], gt[:, 1], label = 'GT')
plt.legend();
plt.grid()
plt.title("Baseline controller step response")

plt.show()