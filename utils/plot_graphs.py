import matplotlib.pyplot as plt

train_accuracies = [
    2.64, 6.67, 16.21, 22.99, 31.91, 41.42, 47.63, 52.88, 58.81, 63.60,
    66.13, 68.96, 72.08, 74.55, 77.14, 78.27, 80.65, 82.14, 83.66, 84.00,
    85.02, 86.53, 87.01, 88.07, 89.01, 89.56, 89.26, 90.38, 90.29, 90.91,
    91.30, 91.37, 92.18, 92.13, 92.55, 93.21, 93.54, 93.84, 93.76, 94.49,
    94.36, 94.62, 94.92, 95.15, 95.31, 95.41, 95.52, 95.65, 95.99, 95.73
]

val_accuracies = [
    2.81, 13.38, 22.37, 29.40, 41.67, 45.55, 57.56, 58.06, 66.76, 69.80,
    74.75, 73.71, 78.53, 81.14, 81.77, 85.02, 85.22, 84.98, 83.61, 87.53,
    87.93, 87.66, 90.57, 86.82, 92.74, 89.50, 93.98, 89.16, 93.21, 90.80,
    93.81, 94.95, 92.71, 95.18, 94.95, 94.41, 95.55, 95.15, 95.79, 94.58,
    95.72, 94.98, 97.02, 97.19, 96.92, 97.46, 96.52, 96.25, 96.62, 97.29
]

val_losses = [
    3.5841, 2.5743, 2.1202, 1.8738, 1.4763, 1.4032, 1.1126, 1.1034, 0.8906, 0.7761,
    0.6807, 0.6910, 0.5926, 0.5196, 0.5033, 0.4040, 0.4142, 0.4438, 0.4794, 0.3479,
    0.3308, 0.3779, 0.2997, 0.4763, 0.2346, 0.3568, 0.1969, 0.3716, 0.2238, 0.3223,
    0.2039, 0.1645, 0.2807, 0.1774, 0.1795, 0.1891, 0.1731, 0.2041, 0.1521, 0.2132,
    0.1569, 0.2091, 0.1218, 0.1426, 0.1235, 0.1233, 0.1429, 0.1652, 0.1342, 0.1279
]


train_losses = [
    3.5888, 3.1960, 2.4428, 2.1150, 1.7945, 1.5585, 1.3976, 1.2692, 1.1211, 1.0154,
    0.9433, 0.8645, 0.7969, 0.7269, 0.6615, 0.6295, 0.5722, 0.5317, 0.5036, 0.4924,
    0.4561, 0.4220, 0.4027, 0.3867, 0.3508, 0.3406, 0.3457, 0.3224, 0.3089, 0.3193,
    0.2838, 0.2922, 0.2734, 0.2719, 0.2617, 0.2430, 0.2419, 0.2371, 0.2320, 0.2089,
    0.2110, 0.2114, 0.2010, 0.1892, 0.1936, 0.1985, 0.1835, 0.1821, 0.1707, 0.1747
]

epochs = range(1, len(train_accuracies) + 1)

# --- Plot Accuracy (HD) ---
plt.figure(figsize=(12, 7), dpi=200)  # Higher DPI for HD
plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange', linewidth=2)
plt.title('Training and Validation Accuracy using Mediapipe and VideoMAE', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot Loss (HD) ---
plt.figure(figsize=(12, 7), dpi=200)  # Higher DPI for HD
plt.plot(epochs, train_losses, label='Training Loss', color='red', linewidth=2)
plt.plot(epochs, val_losses, label='Validation Loss', color='green', linewidth=2)
plt.title('Training and Validation Loss using Mediapipe and VideoMAE', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()