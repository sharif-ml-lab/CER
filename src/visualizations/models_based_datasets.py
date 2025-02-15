import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

mode = "math"  # multihop

model_labels = ['LLaMA-3.1-8B', 'Mistral-7B']

if mode == "math":
    categories = ['GSM8K', 'MATH', 'MathQA']
    blue_values_LLAMA_8B = [90.0, 58.2, 68.2]
    red_values_LLAMA_8B = [70.6, 38.4, 44.8]

    blue_values_MISTRAL_7B = [65.2, 18.0, 22.6]
    red_values_MISTRAL_7B = [28.8, 12.8, 12.4]

else:
    categories = ['TriviaQA', 'HotpotQA']

    blue_values_Trivia = [66.0, 14.4]
    red_values_Trivia = [56.2, 9.2]

    blue_values_Hotpot = [54.4, 10.4]
    red_values_Hotpot = [46.6, 6.2]

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
    ax.bar(x_Model1, blue_values_LLAMA_8B, width, color=blue_color)
    ax.bar(x_Model1 + width, red_values_LLAMA_8B, width, color=red_color)
    ax.bar(x_Model2, blue_values_MISTRAL_7B, width, color=blue_color)
    ax.bar(x_Model2 + width, red_values_MISTRAL_7B, width, color=red_color)
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
    mpatches.Patch(color=blue_color, label='CER (ALL)'),
    mpatches.Patch(color=red_color, label='CER (LAST)')
]
ax.legend(handles=legend_patches, loc='upper center', fontsize=10,
          frameon=False, bbox_to_anchor=(0.5, 1.15), ncol=2)

for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height - 2,
            f'{height:.1f}', ha='center', va='top', color='white', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(f'{mode}_chart.png', dpi=300, bbox_inches='tight')
plt.show()
