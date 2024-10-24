import torch
import numpy as np
import torch.nn as nn

# Remove small values
def remove_small(heatmap, threshold, device):
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).to(device)
    heatmap = torch.where(heatmap < threshold, z, heatmap)
    return heatmap

# Check link
def check_link(min, max, keypoint, device):
    # Define pairs of keypoints based on BODY_25 skeleton
    BODY_25_pairs = np.array([
        [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
        [12, 13], [13, 14], [1, 0], [14, 15], [15, 16], [14, 17], [11, 18], [18, 19], [11, 20]]
    )

    # Output tensor to store results for each frame
    # keypoint.shape = (batch_size, channel, 3)
    keypoint_output = torch.ones(keypoint.shape[0], 20).to(device)

    # Loop through each frame
    for f in range(keypoint.shape[0]):
        # Check each keypoint pair
        for i in range(20):
            a = keypoint[f, BODY_25_pairs[i, 0]]  # First keypoint
            b = keypoint[f, BODY_25_pairs[i, 1]]  # Second keypoint
            s = torch.sum((a - b)**2)  # Compute squared distance

            # Apply penalties if distance is outside allowed range
            if s < min[i]:
                keypoint_output[f, i] = min[i] - s
            elif s > max[i]:
                keypoint_output[f, i] = s - max[i]
            else:
                keypoint_output[f, i] = 0

    return keypoint_output

