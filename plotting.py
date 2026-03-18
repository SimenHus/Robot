import matplotlib.pyplot as plt


with open('./build/output.csv', 'r') as f:
    data = [float(_.split(',')[0]) for _ in f.readlines()]

time = [i for i in range(len(data))]

r = 2
plt.plot(time, data)
plt.plot([time[0], time[-1]], [r, r], '--');
plt.legend(['y', 'r']);
plt.ylabel("Output")
plt.xlabel("Time $t$ [sec]")
plt.title("Baseline controller step response")

plt.show()