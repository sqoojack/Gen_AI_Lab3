import matplotlib.pyplot as plt
import numpy as np

baseline_scores = {
    "ROUGE-L": 0.0350,
    "Correctness": 0.2300,
}

langchain_scores = {
    "ROUGE-L": 0.2186,
    "Correctness": 0.3200,
}

custom_chunk_scores = {
    "ROUGE-L": 0.4193,
    "Correctness": 0.3900,
}

methods = ['baseline', 'custom_chunk', 'langchain']

rouge_l = [baseline_scores["ROUGE-L"], custom_chunk_scores["ROUGE-L"], langchain_scores["ROUGE-L"]]
correctness = [baseline_scores["Correctness"], custom_chunk_scores["Correctness"], langchain_scores["Correctness"]]

x = np.arange(len(methods))
width = 0.35  # 每組的柱狀寬度

plt.figure(figsize=(9, 6))

# ROUGE-L (左)
bars1 = plt.bar(x - width/2, rouge_l, width, label='ROUGE-L', color='skyblue')
# Correctness (右)
bars2 = plt.bar(x + width/2, correctness, width, label='Correctness', color='lightcoral')

plt.xlabel('Method', fontsize=13, fontweight='bold')
plt.ylabel('Score', fontsize=13, fontweight='bold')
plt.title('Comparison of ROUGE-L and Correctness among RAG Methods', fontsize=15, fontweight='bold')
plt.xticks(x, methods, fontsize=12, fontweight='bold')
plt.ylim(0, max(max(rouge_l), max(correctness)) * 1.2)
plt.legend(fontsize=12)

# 在每個 bar 上標註分數
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=11)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('plot_comparison/RAG_technique.png', dpi=300)
plt.show()