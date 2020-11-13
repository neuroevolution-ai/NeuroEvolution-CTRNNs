import numpy as np
import matplotlib.pyplot as plt


old_repo_values = np.loadtxt("/home/pdeubel/PycharmProjects/NeuroEvolution-CTRNNs/episode-times.txt")
new_repo_values = np.loadtxt("/home/pdeubel/PycharmProjects/NeuroEvolution-CTRNN_new/episode_times.txt")

# old_repo_values = old_repo_values[2500:]
# new_repo_values = new_repo_values[2500:]

print("Mean Old {} | Mean New {}".format(np.mean(old_repo_values), np.mean(new_repo_values)))
print("Median Old {} | Median New {}".format(np.median(old_repo_values), np.median(new_repo_values)))
print("Std Old {} | Std New {}".format(np.std(old_repo_values), np.std(new_repo_values)))
print("Max Old {} | Max New {}".format(np.max(old_repo_values), np.max(new_repo_values)))
print("Min Old {} | Min New {}".format(np.min(old_repo_values), np.min(new_repo_values)))

plt.plot(np.arange(new_repo_values.size), new_repo_values, label="New Repo n={}".format(new_repo_values.size))
plt.plot(np.arange(old_repo_values.size), old_repo_values, label="Old Repo n={}".format(old_repo_values.size))


plt.xlabel("Time Measurements")
plt.ylabel("Time for CTRNN.step (s)")
plt.legend()
plt.show()
