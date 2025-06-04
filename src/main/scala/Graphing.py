import matplotlib.pyplot as plt

k_values = list(range(1, 101))
rmse_values = [
    0.9203, 0.8116, 0.7725, 0.7547, 0.7467, 0.7383, 0.7289, 0.7194, 0.7158, 0.7086,
    0.7018, 0.6974, 0.6964, 0.6963, 0.6917, 0.6906, 0.6889, 0.6891, 0.6878, 0.6837,
    0.6810, 0.6803, 0.6783, 0.6752, 0.6748, 0.6733, 0.6728, 0.6726, 0.6708, 0.6699,
    0.6706, 0.6704, 0.6689, 0.6696, 0.6683, 0.6681, 0.6686, 0.6697, 0.6695, 0.6700,
    0.6698, 0.6697, 0.6689, 0.6667, 0.6648, 0.6638, 0.6632, 0.6626, 0.6631, 0.6634,
    0.6638, 0.6649, 0.6648, 0.6659, 0.6656, 0.6647, 0.6638, 0.6643, 0.6641, 0.6627,
    0.6616, 0.6615, 0.6619, 0.6616, 0.6604, 0.6599, 0.6602, 0.6600, 0.6594, 0.6592,
    0.6590, 0.6585, 0.6581, 0.6575, 0.6576, 0.6571, 0.6557, 0.6550, 0.6544, 0.6543,
    0.6540, 0.6538, 0.6534, 0.6543, 0.6542, 0.6542, 0.6540, 0.6531, 0.6528, 0.6529,
    0.6520, 0.6515, 0.6518, 0.6526, 0.6524, 0.6527, 0.6524, 0.6526, 0.6530, 0.6529
]

min_rmse = min(rmse_values)
best_k = k_values[rmse_values.index(min_rmse)]

plt.figure(figsize=(12, 6))
plt.plot(k_values, rmse_values, marker='o', linestyle='-', color='teal', label="RMSE")

plt.scatter([best_k], [min_rmse], color='red', s=80, zorder=5, label=f"Best k = {best_k}")
plt.annotate(f"Best k = {best_k}\nRMSE = {min_rmse:.4f}",
             xy=(best_k, min_rmse),
             xytext=(best_k - 15, min_rmse + 0.01),
             arrowprops=dict(arrowstyle="->", lw=1.5),
             fontsize=10)

plt.title("KNN Hyperparameter Tuning - White Wine (k vs. RMSE)", fontsize=14)
plt.xlabel("k (Number of Neighbors)", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.xticks(range(0, 101, 5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
