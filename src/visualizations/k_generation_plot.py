import matplotlib.pyplot as plt
import numpy as np

# based on LLaMA-3.3-3B on MATH Reasoning Dataset
generations = np.array([3, 5, 10])
cer = np.array([0.4, 0.5, 0.56])
p_true = np.array([0.0, 0.4, 0.442])
pe = np.array([0.1, 0.2, 0.44])
nl = np.array([0.2, 0.3, 0.426])
ne = np.array([0.3, 0.4, 0.5])

plt.figure(figsize=(6, 4))
plt.plot(generations, cer, 'o-', label="CER")
plt.plot(generations, p_true, 'o-', label="P(True)")
plt.plot(generations, pe, 'o-', label="PE")
plt.plot(generations, nl, 'o-', label="NL")
plt.plot(generations, ne, 'o-', label="NE")


plt.xlabel("Number of Generations", fontsize=10)
plt.ylabel("ACCURACY", fontsize=10)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("k_generation.png", dpi=300, bbox_inches="tight")
plt.show()
