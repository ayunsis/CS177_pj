import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('src/sfcnn-origin/input/1a28_grid.pkl', 'rb') as f:
    grid = pickle.load(f)[0]
with open('src/sfcnn-origin/input/1a28_grid.pkl', 'rb') as f:
    heatmap = pickle.load(f)

x = []
y = []
z = []
c = []
for i in range(20):
    for j in range(20):
        for k in range(20):
            tmp = grid[i, j, k]
            if tmp[15:].sum() >= 1:
                x.append(i)
                y.append(j)
                z.append(k)
                n = list(tmp).index(1.0)
                if n in [15, 16, 17]:
                    c.append(1)
                elif n in [18, 19, 20]:
                    c.append(1.1)
                elif n == 21:
                    c.append(1.2)
                else:
                    c.append(0.9)
            elif tmp[1:14].sum() >= 1:
                x.append(i)
                y.append(j)
                z.append(k)
                c.append(0.3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=c, cmap='coolwarm', s=20)

# Handle 5D heatmap: reduce to 3D
heatmap_data = np.array(heatmap)

while heatmap_data.ndim > 3:
    heatmap_data = heatmap_data[0]
print(heatmap_data.sum())
threshold = np.percentile(heatmap_data, 99) # 3 std
hx, hy, hz = np.where(heatmap_data >= threshold)
ax.scatter(hx, hy, hz, alpha=0.1, c='gray', s=1, label='Heatmap')

plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with Heatmap Overlay')
plt.show()