import matplotlib.pyplot as plt
import numpy as np

# based on LLaMA-3.3-3B on MATH Reasoning Dataset
generations = np.array([3, 5, 10])
cer = np.array([0.444, 0.462, 0.560]) # weighted-mean
self_consistency = np.array([0.422, 0.454, 0.512])
p_true = np.array([0.388, 0.396, 0.442])
pe = np.array([0.404, 0.408, 0.440])
# nl = np.array([0.438, 0.464, 0.426])
# ne = np.array([0.386, 0.458, 0.400]) 
ll = np.array([0.428, 0.398, 0.402])

plt.figure(figsize=(8, 5))
plt.plot(generations, cer, 'o-', label="CER")
plt.plot(generations, self_consistency, 'o-', label="Self-Consistency")
plt.plot(generations, p_true, 'o-', label="P(True)")
plt.plot(generations, pe, 'o-', label="PE")
# plt.plot(generations, nl, 'o-', label="NL")
# plt.plot(generations, ne, 'o-', label="NE")
plt.plot(generations, ll, 'o-', label="LL")


plt.xlabel("Number of Generations", fontsize=10)
plt.ylabel("ACCURACY", fontsize=10)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("k_generation.png", dpi=300, bbox_inches="tight")
plt.show()
