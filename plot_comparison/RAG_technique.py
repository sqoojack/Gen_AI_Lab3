# python3 plot_comparison/RAG_technique.py

import matplotlib.pyplot as plt
import numpy as np

baseline_scores = {
    "ROUGE-L": 0.0000,
    "Correctness": 0.2400,
}

langchain_scores = {
    "ROUGE-L": 0.2186,
    "Correctness": 0.3200,
}

custom_chunk_scores = {
    "ROUGE-L": 0.4193,
    "Correctness": 0.3900,
}

methods = ['baseline', 'langchain', 'custom_chunk']

rouge_l = [baseline_scores["ROUGE-L"], langchain_scores["ROUGE-L"], custom_chunk_scores["ROUGE-L"]]
correctness = [baseline_scores["Correctness"], langchain_scores["Correctness"], custom_chunk_scores["Correctness"]]

x = np.arange(len(methods))
width = 0.5

plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
plt.bar(x, rouge_l, width, color='skyblue')
plt.xticks(x, methods, fontweight='bold')    # set x-axis name
plt.ylabel('ROUGE-L', fontweight='bold')
plt.title("Different RAG method's Rouge-L", fontweight='bold')
plt.ylim(0, max(rouge_l) * 1.2)

# Correctness 子圖
plt.subplot(1, 2, 2)
plt.bar(x, correctness, width, color='lightcoral')
plt.xticks(x, methods, fontweight='bold')
plt.ylabel('Correctness', fontweight='bold')
plt.title("Different RAG method's Correctness", fontweight='bold')
plt.ylim(0, max(correctness) * 1.2)

plt.tight_layout()
plt.savefig('plot_comparison/RAG_technique.png', dpi=300)
plt.show()