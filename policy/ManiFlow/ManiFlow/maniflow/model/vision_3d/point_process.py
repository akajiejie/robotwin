# provide some torch/numpy implementations for point cloud processing
# @Yanjie Ze

import torch
import numpy as np
import pytorch3d.ops as torch3d_ops
import random


__all__ = ["shuffle_point_torch", "pad_point_torch", "uniform_sampling_torch", "fps_torch"]

def shuffle_point_numpy(point_cloud):
    B, N, C = point_cloud.shape
    indices = np.random.permutation(N)
    return point_cloud[:, indices]

def pad_point_numpy(point_cloud, num_points):
    B, N, C = point_cloud.shape
    if num_points > N:
        num_pad = num_points - N
        pad_points = np.zeros((B, num_pad, C))
        point_cloud = np.concatenate([point_cloud, pad_points], axis=1)
        point_cloud = shuffle_point_numpy(point_cloud)
    return point_cloud

def uniform_sampling_numpy(point_cloud, num_points):
    B, N, C = point_cloud.shape
    # padd if num_points > N
    if num_points > N:
        return pad_point_numpy(point_cloud, num_points)
    
    # random sampling
    indices = np.random.permutation(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

def shuffle_point_torch(point_cloud):
    B, N, C = point_cloud.shape
    indices = torch.randperm(N)
    return point_cloud[:, indices]

def pad_point_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        num_pad = num_points - N
        pad_points = torch.zeros(B, num_pad, C).to(device)
        point_cloud = torch.cat([point_cloud, pad_points], dim=1)
        point_cloud = shuffle_point_torch(point_cloud)
    return point_cloud

def uniform_sampling_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    # padd if num_points > N
    if num_points == N:
        return point_cloud
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)
    
    # random sampling
    indices = torch.randperm(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points



def fps_torch(points, num_points=1024):
    """
    points: (B, N, C)
    num_points: int
    
    return:
    sampled_points: (B, num_points, C)
    indices: (B, num_points)
    """
    B, N, C = points.shape
    K = num_points
    if C == 3:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points, K=K)
        sampled_points = sampled_points.contiguous()
    elif C > 3:
        # pcd + features, pcd must be the first 3 channels
        _, indices = torch3d_ops.sample_farthest_points(points=points[..., :3], K=K)
        sampled_points = points[torch.arange(B).unsqueeze(1), indices]
   
    return sampled_points, indices

def fps_openpoint(points, num_points=1024):
    """
    points: (B, N, C)
    num_points: int
    
    return:
    sampled_points: (B, num_points, C)
    indices: (B, num_points)
    """
    B, N, C = points.shape
    K = num_points
    if C == 3:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points, K=K)
        sampled_points = sampled_points.contiguous()
    elif C > 3:
        # pcd + features, pcd must be the first 3 channels
        _, indices = torch3d_ops.sample_farthest_points(points=points[..., :3], K=K)
        sampled_points = points[torch.arange(B).unsqueeze(1), indices]
   
    return sampled_points

class PointCloudColorJitterSingle:
    """ColorJitter-like transform for point clouds with RGB colors.
    
    Args:
        brightness: How much to jitter brightness (float or tuple of floats)
        contrast: How much to jitter contrast (float or tuple of floats)  
        saturation: How much to jitter saturation (float or tuple of floats)
        hue: How much to jitter hue (float or tuple of floats), should be in [-0.5, 0.5]
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness', center=1.0)
        self.contrast = self._check_input(contrast, 'contrast', center=1.0)
        self.saturation = self._check_input(saturation, 'saturation', center=1.0)
        self.hue = self._check_input(hue, 'hue', center=0.0, bound=(-0.5, 0.5))

    @staticmethod
    def _check_input(value, name, center=1.0, bound=None):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} should be non-negative")
            if name == 'hue':
                # Hue is special case - symmetric around 0
                value = [-value, value]
                if bound is not None:
                    value = [max(value[0], bound[0]), min(value[1], bound[1])]
            else:
                # Brightness/contrast/saturation are multiplicative
                value = [max(0, center - value), center + value]
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if bound is not None:
                value = [max(value[0], bound[0]), min(value[1], bound[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2")
        return value

    def __call__(self, pointcloud):
        """
        Args:
            pointcloud: (N, C) tensor where C >= 6 and last 3 channels are RGB colors in [0, 1]
        
        Returns:
            pointcloud with augmented colors
        """
        if pointcloud.shape[1] < 6:
            return pointcloud  # No color channels to augment
            
        # Extract colors (last 3 channels assumed to be RGB)
        colors = pointcloud[..., -3:].clone()  # (N, 3)
        
        # Apply transformations in order
        colors = self._adjust_brightness(colors)
        colors = self._adjust_contrast(colors)
        colors = self._adjust_saturation(colors)
        colors = self._adjust_hue(colors)
        
        # Clamp to valid range and update pointcloud
        colors = torch.clamp(colors, 0, 1)
        pointcloud = pointcloud.clone()
        pointcloud[..., -3:] = colors
        
        return pointcloud

    def _adjust_brightness(self, colors):
        """Apply brightness adjustment (multiplicative, same as torchvision)"""
        if self.brightness[0] == self.brightness[1]:
            return colors
        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        return colors * brightness_factor

    def _adjust_contrast(self, colors):
        """Apply contrast adjustment (same as torchvision)"""
        if self.contrast[0] == self.contrast[1]:
            return colors
        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        # Contrast around 0.5 (middle gray) like torchvision
        return (colors - 0.5) * contrast_factor + 0.5

    def _adjust_saturation(self, colors):
        """Apply saturation adjustment"""
        if self.saturation[0] == self.saturation[1]:
            return colors
        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        
        # Convert to grayscale using standard RGB weights
        gray = (colors * torch.tensor([0.299, 0.587, 0.114])).sum(dim=-1, keepdim=True)
        gray = gray.expand_as(colors)
        
        # Blend between original and grayscale
        return colors * saturation_factor + gray * (1 - saturation_factor)

    def _adjust_hue(self, colors):
        """Apply hue adjustment using HSV color space"""
        if self.hue[0] == self.hue[1]:
            return colors
        hue_shift = random.uniform(self.hue[0], self.hue[1])
        
        if abs(hue_shift) < 1e-6:
            return colors
            
        # Convert RGB to HSV
        hsv = self._rgb_to_hsv(colors)
        
        # Adjust hue
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
        
        # Convert back to RGB
        return self._hsv_to_rgb(hsv)

    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        max_val, max_idx = torch.max(rgb, dim=-1)
        min_val, _ = torch.min(rgb, dim=-1)
        diff = max_val - min_val
        
        # Hue calculation
        h = torch.zeros_like(max_val)
        mask = diff != 0
        
        # Red is max
        red_mask = mask & (max_idx == 0)
        h[red_mask] = ((g[red_mask] - b[red_mask]) / diff[red_mask]) / 6.0
        
        # Green is max  
        green_mask = mask & (max_idx == 1)
        h[green_mask] = ((b[green_mask] - r[green_mask]) / diff[green_mask] + 2) / 6.0
        
        # Blue is max
        blue_mask = mask & (max_idx == 2)
        h[blue_mask] = ((r[blue_mask] - g[blue_mask]) / diff[blue_mask] + 4) / 6.0
        
        # Normalize hue to [0, 1]
        h = h % 1.0
        
        # Saturation
        s = torch.where(max_val != 0, diff / max_val, torch.zeros_like(max_val))
        
        # Value
        v = max_val
        
        return torch.stack([h, s, v], dim=-1)

    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB color space"""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        
        h = h * 6.0  # Scale to [0, 6)
        i = torch.floor(h).long()
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        # Initialize RGB
        rgb = torch.zeros_like(hsv)
        
        # Case 0: R=v, G=t, B=p
        mask0 = (i % 6 == 0)
        rgb[mask0] = torch.stack([v[mask0], t[mask0], p[mask0]], dim=-1)
        
        # Case 1: R=q, G=v, B=p  
        mask1 = (i % 6 == 1)
        rgb[mask1] = torch.stack([q[mask1], v[mask1], p[mask1]], dim=-1)
        
        # Case 2: R=p, G=v, B=t
        mask2 = (i % 6 == 2) 
        rgb[mask2] = torch.stack([p[mask2], v[mask2], t[mask2]], dim=-1)
        
        # Case 3: R=p, G=q, B=v
        mask3 = (i % 6 == 3)
        rgb[mask3] = torch.stack([p[mask3], q[mask3], v[mask3]], dim=-1)
        
        # Case 4: R=t, G=p, B=v
        mask4 = (i % 6 == 4)
        rgb[mask4] = torch.stack([t[mask4], p[mask4], v[mask4]], dim=-1)
        
        # Case 5: R=v, G=p, B=q
        mask5 = (i % 6 == 5)
        rgb[mask5] = torch.stack([v[mask5], p[mask5], q[mask5]], dim=-1)
        
        return rgb


# # Usage example - drop-in replacement for your ColorJitter setup:
# if self.aug_color:
#     pc_jitter = PointCloudColorJitter(
#         brightness=self.aug_color_params[0],
#         contrast=self.aug_color_params[1], 
#         saturation=self.aug_color_params[2],
#         hue=self.aug_color_params[3]
#     )
#     self.jitter = T.RandomApply([pc_jitter], p=self.aug_color_prob)



class PointCloudColorJitter:
    """ColorJitter-like transform for point clouds with RGB colors.
    Supports both single point clouds (N, C) and batched point clouds (B, N, C).
    
    Args:
        brightness: How much to jitter brightness (float or tuple of floats)
        contrast: How much to jitter contrast (float or tuple of floats)  
        saturation: How much to jitter saturation (float or tuple of floats)
        hue: How much to jitter hue (float or tuple of floats), should be in [-0.5, 0.5]
        same_on_batch: If True, apply same augmentation to all items in batch (default: False)
        aug_prob: Probability of augmenting each individual point cloud in batch (default: 0.2)
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, same_on_batch=False, aug_prob=0.2):
        self.brightness = self._check_input(brightness, 'brightness', center=1.0)
        self.contrast = self._check_input(contrast, 'contrast', center=1.0)
        self.saturation = self._check_input(saturation, 'saturation', center=1.0)
        self.hue = self._check_input(hue, 'hue', center=0.0, bound=(-0.5, 0.5))
        self.same_on_batch = same_on_batch
        self.aug_prob = aug_prob

    @staticmethod
    def _check_input(value, name, center=1.0, bound=None):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} should be non-negative")
            if name == 'hue':
                # Hue is special case - symmetric around 0
                value = [-value, value]
                if bound is not None:
                    value = [max(value[0], bound[0]), min(value[1], bound[1])]
            else:
                # Brightness/contrast/saturation are multiplicative
                value = [max(0, center - value), center + value]
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if bound is not None:
                value = [max(value[0], bound[0]), min(value[1], bound[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2")
        return value

    def __call__(self, pointcloud):
        """
        Args:
            pointcloud: (N, C) or (B, N, C) tensor where C >= 6 and last 3 channels are RGB colors in [0, 1]
        
        Returns:
            pointcloud with augmented colors, same shape as input
        """
        if pointcloud.dim() not in [2, 3]:
            raise ValueError("pointcloud should be 2D (N, C) or 3D (B, N, C)")
            
        if pointcloud.shape[-1] < 6:
            return pointcloud  # No color channels to augment
        
        is_batched = pointcloud.dim() == 3
        if not is_batched:
            # Add batch dimension for uniform processing
            pointcloud = pointcloud.unsqueeze(0)
        
        batch_size = pointcloud.shape[0]
        
        # Generate augmentation mask based on aug_prob
        if self.aug_prob < 1.0:
            aug_mask = torch.rand(batch_size, device=pointcloud.device) < self.aug_prob
        else:
            aug_mask = torch.ones(batch_size, dtype=torch.bool, device=pointcloud.device)
        
        # Extract colors (last 3 channels assumed to be RGB)
        colors = pointcloud[..., -3:].clone()  # (B, N, 3)
        original_colors = colors.clone()  # Keep original for non-augmented items
        
        # Apply transformations in order (only if at least one item needs augmentation)
        if aug_mask.any():
            colors = self._adjust_brightness(colors, batch_size, aug_mask)
            colors = self._adjust_contrast(colors, batch_size, aug_mask)
            colors = self._adjust_saturation(colors, batch_size, aug_mask)
            colors = self._adjust_hue(colors, batch_size, aug_mask)
        
        # Keep original colors for non-augmented items
        if not aug_mask.all():
            colors[~aug_mask] = original_colors[~aug_mask]
        
        # Clamp to valid range and update pointcloud
        colors = torch.clamp(colors, 0, 1)
        pointcloud = pointcloud.clone()
        pointcloud[..., -3:] = colors
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            pointcloud = pointcloud.squeeze(0)
        
        return pointcloud

    def _sample_factors(self, param_range, batch_size, aug_mask=None):
        """Sample random factors for batch"""
        if param_range[0] == param_range[1]:
            return torch.full((batch_size,), param_range[0])
        
        if self.same_on_batch:
            factor = random.uniform(param_range[0], param_range[1])
            factors = torch.full((batch_size,), factor)
        else:
            # Different factor for each batch item
            factors = torch.FloatTensor(batch_size).uniform_(param_range[0], param_range[1])
        
        # For non-augmented items, set factors to identity values
        if aug_mask is not None and not aug_mask.all():
            if param_range == self.hue:
                # Hue identity is 0
                factors[~aug_mask] = 0.0
            else:
                # Brightness/contrast/saturation identity is 1
                factors[~aug_mask] = 1.0
                
        return factors

    def _adjust_brightness(self, colors, batch_size, aug_mask=None):
        """Apply brightness adjustment (multiplicative, same as torchvision)"""
        brightness_factors = self._sample_factors(self.brightness, batch_size, aug_mask)
        
        # Reshape for broadcasting: (B, 1, 1)
        brightness_factors = brightness_factors.view(batch_size, 1, 1).to(colors.device)
        return colors * brightness_factors

    def _adjust_contrast(self, colors, batch_size, aug_mask=None):
        """Apply contrast adjustment (same as torchvision)"""
        contrast_factors = self._sample_factors(self.contrast, batch_size, aug_mask)
        
        # Reshape for broadcasting: (B, 1, 1)
        contrast_factors = contrast_factors.view(batch_size, 1, 1).to(colors.device)
        
        # Contrast around 0.5 (middle gray) like torchvision
        return (colors - 0.5) * contrast_factors + 0.5

    def _adjust_saturation(self, colors, batch_size, aug_mask=None):
        """Apply saturation adjustment"""
        saturation_factors = self._sample_factors(self.saturation, batch_size, aug_mask)
        
        # Reshape for broadcasting: (B, 1, 1)
        saturation_factors = saturation_factors.view(batch_size, 1, 1).to(colors.device)
        
        # Convert to grayscale using standard RGB weights
        rgb_weights = torch.tensor([0.299, 0.587, 0.114]).to(colors.device)
        gray = (colors * rgb_weights.view(1, 1, 3)).sum(dim=-1, keepdim=True)
        gray = gray.expand_as(colors)
        
        # Blend between original and grayscale
        return colors * saturation_factors + gray * (1 - saturation_factors)

    def _adjust_hue(self, colors, batch_size, aug_mask=None):
        """Apply hue adjustment using HSV color space"""
        hue_shifts = self._sample_factors(self.hue, batch_size, aug_mask)
        
        # Check if any hue shift is significant
        if torch.all(torch.abs(hue_shifts) < 1e-6):
            return colors
        
        # Reshape for broadcasting: (B, 1, 1)
        hue_shifts = hue_shifts.view(batch_size, 1, 1).to(colors.device)
        
        # Convert RGB to HSV
        hsv = self._rgb_to_hsv(colors)
        
        # Adjust hue
        hsv[..., 0:1] = (hsv[..., 0:1] + hue_shifts) % 1.0
        
        # Convert back to RGB
        return self._hsv_to_rgb(hsv)

    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space (batched)"""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        max_val, max_idx = torch.max(rgb, dim=-1)
        min_val, _ = torch.min(rgb, dim=-1)
        diff = max_val - min_val
        
        # Hue calculation
        h = torch.zeros_like(max_val)
        mask = diff != 0
        
        # Red is max
        red_mask = mask & (max_idx == 0)
        h[red_mask] = ((g[red_mask] - b[red_mask]) / diff[red_mask]) / 6.0
        
        # Green is max  
        green_mask = mask & (max_idx == 1)
        h[green_mask] = ((b[green_mask] - r[green_mask]) / diff[green_mask] + 2) / 6.0
        
        # Blue is max
        blue_mask = mask & (max_idx == 2)
        h[blue_mask] = ((r[blue_mask] - g[blue_mask]) / diff[blue_mask] + 4) / 6.0
        
        # Normalize hue to [0, 1]
        h = h % 1.0
        
        # Saturation
        s = torch.where(max_val != 0, diff / max_val, torch.zeros_like(max_val))
        
        # Value
        v = max_val
        
        return torch.stack([h, s, v], dim=-1)

    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB color space (batched)"""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        
        h = h * 6.0  # Scale to [0, 6)
        i = torch.floor(h).long()
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        # Initialize RGB
        rgb = torch.zeros_like(hsv)
        
        # Case 0: R=v, G=t, B=p
        mask0 = (i % 6 == 0)
        rgb[mask0] = torch.stack([v[mask0], t[mask0], p[mask0]], dim=-1)
        
        # Case 1: R=q, G=v, B=p  
        mask1 = (i % 6 == 1)
        rgb[mask1] = torch.stack([q[mask1], v[mask1], p[mask1]], dim=-1)
        
        # Case 2: R=p, G=v, B=t
        mask2 = (i % 6 == 2) 
        rgb[mask2] = torch.stack([p[mask2], v[mask2], t[mask2]], dim=-1)
        
        # Case 3: R=p, G=q, B=v
        mask3 = (i % 6 == 3)
        rgb[mask3] = torch.stack([p[mask3], q[mask3], v[mask3]], dim=-1)
        
        # Case 4: R=t, G=p, B=v
        mask4 = (i % 6 == 4)
        rgb[mask4] = torch.stack([t[mask4], p[mask4], v[mask4]], dim=-1)
        
        # Case 5: R=v, G=p, B=q
        mask5 = (i % 6 == 5)
        rgb[mask5] = torch.stack([v[mask5], p[mask5], q[mask5]], dim=-1)
        
        return rgb


# # Usage examples:
# if __name__ == "__main__":
#     # Test with single point cloud
#     pc_single = torch.rand(1000, 6)  # N=1000, C=6 (xyz + rgb)
    
#     # Test with batched point clouds  
#     pc_batch = torch.rand(4, 1000, 6)  # B=4, N=1000, C=6
    
#     # Create jitter transform with 50% augmentation probability
#     jitter = PointCloudColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2, 
#         hue=0.1,
#         aug_prob=0.5  # Only 50% of point clouds in batch will be augmented
#     )
    
#     # Apply to single point cloud
#     augmented_single = jitter(pc_single)
#     print(f"Single: {pc_single.shape} -> {augmented_single.shape}")
    
#     # Apply to batched point clouds (50% chance per item, different augmentation per batch item)
#     augmented_batch = jitter(pc_batch)
#     print(f"Batch: {pc_batch.shape} -> {augmented_batch.shape}")
    
#     # Apply same augmentation to all batch items, but only to 70% of them
#     jitter_same = PointCloudColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2,
#         hue=0.1,
#         same_on_batch=True,
#         aug_prob=0.7
#     )
#     augmented_same = jitter_same(pc_batch)
#     print(f"Same on batch (70% prob): {pc_batch.shape} -> {augmented_same.shape}")
    
#     # Always augment (equivalent to original behavior)
#     jitter_always = PointCloudColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2,
#         hue=0.1,
#         aug_prob=1.0  # Always augment (default)
#     )
#     augmented_always = jitter_always(pc_batch)
#     print(f"Always augment: {pc_batch.shape} -> {augmented_always.shape}")


# test fps_torch
if __name__ == "__main__":
    points = torch.rand(6, 6000, 6).cuda()
    sampled_points, indices = fps_torch(points, 1024)
    print(sampled_points.shape)
    print(indices.shape)
    print(indices)
    print(indices.max())
    print(indices.min())
    print(indices.unique().shape)
    print(indices.unique())
    print(indices.unique().shape)