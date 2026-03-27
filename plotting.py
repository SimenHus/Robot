import matplotlib.pyplot as plt
import numpy as np

with open('./build/kalmanOutput.csv', 'r') as f:
    kalmanData = []
    for i, line in enumerate(f.readlines()):
        items = line.split(',')
        kalmanData.append([float(items[0]), float(items[1])])
    kalmanData = np.array(kalmanData)

with open('./build/groundTruth.csv', 'r') as f:
    gt = []
    for i, line in enumerate(f.readlines()):
        items = line.split(',')
        gt.append([float(items[0]), float(items[1])])
    gt = np.array(gt)

with open('./build/particleOutput.csv', 'r') as f:
    particleData = []
    for i, line in enumerate(f.readlines()):
        items = line.split(',')
        particleData.append([float(items[0]), float(items[1])])
    particleData = np.array(particleData)

time = [i for i in range(len(gt))]
plt.plot(kalmanData[:, 0], kalmanData[:, 1], label = 'Kalman')
plt.plot(particleData[:, 0], particleData[:, 1], label = 'Particles')
plt.plot(gt[:, 0], gt[:, 1], label = 'GT')
plt.legend();
plt.grid()
plt.title("Baseline controller step response")

plt.show()