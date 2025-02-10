import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

mode = "multihop"  # multihop

model_labels = ['OLMo-2-7B', 'LLAMA-3.3-3B']

if mode == "math":
    categories = ['GSM8K', 'MATH', 'MATHQA']
    blue_values_OLMo_2_7B = [11.2, 16.8, 38.5]
    red_values_OLMo_2_7B = [26.4, 48.9, 68.3]

    blue_values_LLAMA = [12.4, 18.2, 40.1]
    red_values_LLAMA = [28.7, 50.3, 70.2]

else:
    categories = ['TriviaQA', 'HotpotQA']

    blue_values_Trivia = [30.1, 50.2]
    red_values_Trivia = [45.3, 65.7]

    blue_values_Hotpot = [32.4, 52.8]
    red_values_Hotpot = [47.6, 67.5]

blue_color = '#1f77b4'
red_color = '#d62728'

x = np.arange(len(categories))
width = 0.3

if mode == "math":
    gap = 2.7
else:
    gap = 1.7

x_Model1 = x - width / 2
x_Model2 = x + width / 2 + gap

fig, ax = plt.subplots(figsize=(20, 6))

for spine in ax.spines.values():
    spine.set_visible(False)

for y in range(0, 101, 20):
    ax.axhline(y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

ax.axvline(x[-1] + 0.5, color='black',
           linestyle=(0, (5, 5)), linewidth=3, alpha=1)

if mode == "math":
    ax.bar(x_Model1, blue_values_OLMo_2_7B, width, color=blue_color)
    ax.bar(x_Model1 + width, red_values_OLMo_2_7B, width, color=red_color)
    ax.bar(x_Model2, blue_values_LLAMA, width, color=blue_color)
    ax.bar(x_Model2 + width, red_values_LLAMA, width, color=red_color)
else:
    ax.bar(x_Model1, blue_values_Trivia, width, color=blue_color)
    ax.bar(x_Model1 + width, red_values_Trivia, width, color=red_color)
    ax.bar(x_Model2, blue_values_Hotpot, width, color=blue_color)
    ax.bar(x_Model2 + width, red_values_Hotpot, width, color=red_color)

ax.set_xticks(range(len(categories + categories)))
ax.set_xticklabels(categories + categories, fontsize=10)

ax.set_ylim(0, 110)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_yticklabels([f"{y:.1f}" for y in np.arange(
    0, 101, 20)], fontweight='bold', fontsize=12)

ax.text(np.mean(x_Model1), 105,
        model_labels[0], ha='center', fontsize=14, fontweight='bold', color='black')
ax.text(np.mean(x_Model2), 105,
        model_labels[1], ha='center', fontsize=14, fontweight='bold', color='black')

legend_patches = [
    mpatches.Patch(color=blue_color, label='Self-Consistency'),
    mpatches.Patch(color=red_color, label='CER')
]
ax.legend(handles=legend_patches, loc='upper center', fontsize=10,
          frameon=False, bbox_to_anchor=(0.5, 1.15), ncol=2)

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height - 5,
            f'{height:.1f}', ha='center', va='top', color='white', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(f'{mode}_chart.png', dpi=300, bbox_inches='tight')
plt.show()
