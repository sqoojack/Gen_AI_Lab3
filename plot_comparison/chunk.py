
# Commandline: python3 plot_comparison/chunk.py
import matplotlib.pyplot as plt
import numpy as np


chunk_size400 = {
    "ROUGE-L": 0.1647,
    "Correctness": 0.2100,
}

chunk_size600 = {
    "ROUGE-L": 0.2186,
    "Correctness": 0.3200,
}

chunk_size800 = {
    "ROUGE-L": 0.2049,
    "Correctness": 0.2700
}

chunk_size1000 = {
    "ROUGE-L": 0.1936,
    "Correctness": 0.3100,
}

chunk_size1200 = {
    "ROUGE-L": 0.1911,
    "Correctness": 0.2900,
}

chunk_sizes = [400, 600, 800, 1000, 1200]
scores = [chunk_size400, chunk_size600, chunk_size800, chunk_size1000, chunk_size1200]

rouge_l = [s["ROUGE-L"] for s in scores]
correctness = [s["Correctness"] for s in scores]
x = np.arange(len(chunk_sizes))
width = 0.5

plt.figure(figsize=(8, 5))

# ROUGE-L subgraph
plt.subplot(1, 2, 1)
plt.bar(x, rouge_l, width, color='skyblue')
plt.xticks(x, chunk_sizes, fontweight='bold')
plt.xlabel('Chunk size', fontweight='bold')
plt.ylabel('ROUGE-L', fontweight='bold')
plt.title("Different Chunk Size's ROUGE-L", fontweight='bold')
plt.ylim(0, max(rouge_l) * 1.2)

# Correctness subgraph
plt.subplot(1, 2, 2)
plt.bar(x, correctness, width, color='lightcoral')
plt.xticks(x, chunk_sizes, fontweight='bold')
plt.xlabel('Chunk size', fontweight='bold')
plt.ylabel('Correctness', fontweight='bold')
plt.title("Different Chunk Size's Correctness", fontweight='bold')
plt.ylim(0, max(correctness) * 1.2)

plt.tight_layout()
plt.savefig('plot_comparison/Chunksize_comparison.png', dpi=300)
plt.show()