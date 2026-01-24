import torch
import numpy as np

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

# Joint indices (from previous analysis)
joint_triplets = [
    [51, 52, 53], [54, 55, 56], [57, 58, 59],
    [108, 109, 110], [111, 112, 113], [114, 115, 116],
    [220, 221, 222], [223, 224, 225], [226, 227, 228],
    [229, 230, 231], [232, 233, 234], [235, 236, 237],
    [238, 239, 240], [241, 242, 243], [244, 245, 246],
    [247, 248, 249], [250, 251, 252], [253, 254, 255],
    [259, 260, 261], [262, 263, 264]
]

# Extract joint positions at frame 0
print("Joint positions at frame 0:\n")
joints_pos = []
for idx, triplet in enumerate(joint_triplets):
    x = data[0, triplet[0]].item()
    y = data[0, triplet[1]].item()
    z = data[0, triplet[2]].item()
    joints_pos.append([x, y, z])
    print(f"Joint {idx:2d}: ({x:6.3f}, {y:6.3f}, {z:6.3f})")

# Calculate pairwise distances to understand structure
joints_arr = np.array(joints_pos)

print("\n\nAnalyzing joint relationships (distances at frame 0):")
print("Looking for nearby joints that might be connected...\n")

# Find closest neighbors for each joint
for i in range(len(joints_arr)):
    distances = []
    for j in range(len(joints_arr)):
        if i != j:
            dist = np.linalg.norm(joints_arr[i] - joints_arr[j])
            distances.append((j, dist))
    
    distances.sort(key=lambda x: x[1])
    closest_3 = distances[:3]
    
    print(f"Joint {i:2d} closest neighbors: ", end="")
    for neighbor_idx, dist in closest_3:
        print(f"Joint {neighbor_idx:2d} (dist={dist:.3f})  ", end="")
    print()

# Analyze vertical positions to identify body parts
print("\n\nJoint heights (Y-axis) to identify body parts:")
y_values = joints_arr[:, 1]
sorted_indices = np.argsort(y_values)

print("From lowest to highest:")
for idx in sorted_indices:
    print(f"Joint {idx:2d}: Y = {y_values[idx]:.3f}")

