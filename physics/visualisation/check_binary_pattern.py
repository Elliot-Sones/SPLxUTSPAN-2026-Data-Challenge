import torch

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')
feat336 = data[:, 336]

print("Feature 336 values (all 104 frames):")
for i in range(len(feat336)):
    print(f"Frame {i:3d}: {feat336[i].item()}")

# Find transitions
print("\n\nTransitions:")
for i in range(len(feat336)-1):
    if feat336[i] != feat336[i+1]:
        print(f"Frame {i} → {i+1}: {feat336[i].item()} → {feat336[i+1].item()}")

