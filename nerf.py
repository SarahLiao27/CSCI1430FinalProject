import torch
import torch.nn as nn
import math
import os
import json
import imageio
import numpy as np
import random

def load_synthetic_images_and_camera_metadata(data_dir, split='train', num_classes=1):
    """
    Load images and camera data from NeRF synthetic dataset format.
    Params:
    - class_dir: Directory for the dataset 
    - split: Train, Test, or val
    - num_classes: Number of class folders to use
    Returns:
    - images: Tensor of shape [N, H, W, 3]
    - poses: Tensor of shape [N, 4, 4]
    - camera_angle_x: Field of view in the x-direction
    """
    all_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    selected_classes = all_classes[:num_classes]  # Take first N classes
    images = []
    poses = []
    camera_angle_x = None
    for cls in selected_classes:
        class_dir = os.path.join(data_dir, cls)
        json_path = os.path.join(class_dir, f'transforms_{split}.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        # Store camera angle (same for all images in class)
        if camera_angle_x is None:
            camera_angle_x = metadata['camera_angle_x']
        elif not math.isclose(camera_angle_x, metadata['camera_angle_x']):
            print(f"Warning: Different camera_angle_x in {cls}")

        for frame in metadata['frames']:
            # Handle different path formats (some end with .png, some don't)
            file_path = frame['file_path']
            if not file_path.endswith('.png'):
                file_path += '.png'
            # Handle both cases where images are in split folder or not
            img_path = os.path.join(class_dir, file_path)
            if not os.path.exists(img_path):
                # Try looking in the split subdirectory
                img_path = os.path.join(class_dir, split, os.path.basename(file_path))
            image = imageio.v2.imread(img_path).astype(np.float32) / 255.0
            transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
            images.append(image)
            poses.append(transform_matrix)
    if not images:
        raise RuntimeError(f"No images found in {data_dir} for split {split}")
    images = torch.tensor(np.array(images))
    poses = torch.tensor(np.array(poses))

    return (images, poses, camera_angle_x)

def generate_camera_rays(camera_pose, H=800, W=800, camera_angle_x=0.6911112070083618):
    """
    Generates the ray origins and directions for all pixels in an image.
    Params:
    - camera_pose: [4, 4] tensor, the transformation matrix
    - H, W: The image height and width
    - camera_angle_x: Field of view in the x-direction
    Returns:
    - rays_o: [H*W, 3] tensor of ray origins (same for all pixels becuase the origin is the camera)
    - rays_d: [H*W, 3] tensor of ray directions (world space)
    """
    # get the device (cpu/gpu) the camera pose is on so we compute everything in the same place
    device = camera_pose.device 
    if isinstance(camera_angle_x, float):
        camera_angle_x = torch.tensor(camera_angle_x, device=device)
    focal = 0.5 * W / torch.tan(camera_angle_x * 0.5)
    # Generate the pixel grid coordinates and convert them to camera space directions
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device), torch.arange(H, dtype=torch.float32, device=device), indexing='xy')
    x = (i - W * 0.5) / focal
    y = -(j - H * 0.5) / focal
    z = -torch.ones_like(x) 
    dirs = torch.stack([x, y, z], dim=-1)
    # Transform ray directions from camera space to world space then normalize
    ray_directions = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    # Get the ray origins from camera position and reshapes the outputs
    ray_origins = camera_pose[:3, 3].expand(ray_directions.shape)  
    rays_o = ray_origins.reshape(-1, 3)
    rays_d = ray_directions.reshape(-1, 3)

    return rays_o, rays_d

def positional_encoding(inputs, num_freqs):
    """
    Converts raw input positions to higher-dimensional representations using sinusoidal encoding.
    Params:
    - inputs: Tensor of input positions (e.g., x, y, z coordinates)
    - num_freqs: Number of frequency bands to use for encoding
    Returns:
    - encoded: Higher-dimensional encoding that helps model high-frequency details
    """
    # Generate the sinusoidal encodings for each frequency band
    encodings = []
    for i in range(num_freqs):
        freq = 2 ** i * math.pi
        encodings.append(torch.sin(freq * inputs))
        encodings.append(torch.cos(freq * inputs))
    return torch.cat(encodings, dim=-1)

class NeRFModel(nn.Module):
    """
    An MLP that represents the neural radiance field (NeRF).
    Takes 3D spatial coordinates and viewing directions and returns density and RGB color.
    """

    def __init__(self, num_frequencies, input_pos_dimensions, input_dir_dimensions):
        """
        initialize the NeRF model architecture including hidden layers,
        activation functions, and output heads for density and color.
        """
        super(NeRFModel, self).__init__()

        self.num_frequencies = num_frequencies
        self.pos_dimensions = input_pos_dimensions * 2 * num_frequencies
        self.dir_dimensions = input_dir_dimensions * 2 * num_frequencies
        hidden_size = 256

        self.pts_linears = nn.ModuleList([
            nn.Linear(self.pos_dimensions, hidden_size),
            nn.Linear(hidden_size, hidden_size),       
            nn.Linear(hidden_size, hidden_size),    
            nn.Linear(hidden_size, hidden_size),   
            #skip connection layer, a bit confusing but tldr we need skip layers for vanishing gradiant problem         
            nn.Linear(hidden_size + self.pos_dimensions, hidden_size),   
            nn.Linear(hidden_size, hidden_size),                
            nn.Linear(hidden_size, hidden_size),          
            nn.Linear(hidden_size, hidden_size)      
        ])
        self.relu = nn.ReLU()
        self.fc_sigma = nn.Linear(hidden_size, 1)
        self.fc_feature = nn.Linear(hidden_size, hidden_size)
        self.view_linear = nn.Linear(hidden_size + self.dir_dimensions, 128)
        self.fc_rgb = nn.Linear(128, 3)

    def forward(self, positions, view_directions):
        """
        Compute density and color values from given 3D positions and view directions.
        """
        pos_encoded = positional_encoding(positions, self.num_frequencies)
        dir_encoded = positional_encoding(view_directions, self.num_frequencies)
        h = pos_encoded
        for i, layer in enumerate(self.pts_linears):
            if i == 4: #for skip connection layer
                h = torch.cat([h, pos_encoded], dim=-1)
            h = self.relu(layer(h))
        sigma = self.fc_sigma(h)
        features = self.relu(self.fc_feature(h)) 
        h_dir = self.relu(self.view_linear(torch.cat([features, dir_encoded], dim=-1)))
        rgb = torch.sigmoid(self.fc_rgb(h_dir))
        output = torch.cat([sigma, rgb], dim=-1)  
        return output

def sample_pdf(bins, weights, N_samples, device):
    """
    Hierarchical sampling using inverse transform sampling.
    Params:
    - bins: [N_rays, N_samples-1], bin edges
    - weights: [N_rays, N_samples-1], weight for each bin
    - N_samples: The number of samples to draw
    - device: The device we are using
    Returns:
    - samples: [N_rays, N_samples] newly sampled depths
    """
    # Calculates the PDF and CDF from the weights
    weights += 1e-5  # Avoid nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    u = torch.rand(cdf.shape[0], N_samples, device=device)
    # Find locations of samples through inverse CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    # Gets the CDF and bin values for interpolation
    inds_g = torch.stack([below, above], -1)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)
    # Do linear interpolation to get final samples
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def render_rays(model, ray_origins, ray_directions, N_coarse=64, N_fine=128, near=2.0, far=10.0, device = 'cpu'):
    """
    Render rays using NeRF with hierarchical sampling.
    Params:
    - model: Our NeRf model
    - ray_origins: Tensor of ray origin positions
    - ray_directions: Tensor of ray direction vectors
    - N_coarse: Number of samples for coarse sampling
    - N_fine: Number of samples for fine sampling
    - near: Near bound for sampling
    - far: Far bound for sampling
    - device: Device being used
    Returns:
    - rgb: Tensor of rendered colors
    """
    N_rays = ray_origins.shape[0]
    # Coarse sampling
    z_vals_coarse = torch.linspace(near, far, N_coarse, device=device).expand(N_rays, N_coarse)
    pts_coarse = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * z_vals_coarse.unsqueeze(-1)
    pts_coarse_flat = pts_coarse.reshape(-1, 3)
    dirs_coarse_flat = ray_directions.unsqueeze(1).expand(-1, N_coarse, -1).reshape(-1, 3)
    with torch.no_grad():  # Reduce memory during coarse pass
        outputs_coarse = model(pts_coarse_flat, dirs_coarse_flat)
    sigma_coarse = outputs_coarse[:, 0].reshape(N_rays, N_coarse)
    rgb_coarse = outputs_coarse[:, 1:].reshape(N_rays, N_coarse, 3)
    # Alpha compositing
    deltas = z_vals_coarse[:, 1:] - z_vals_coarse[:, :-1]
    delta_inf = torch.full((N_rays, 1), 1e10, device=device)
    deltas = torch.cat([deltas, delta_inf], dim=1)
    alpha = 1 - torch.exp(-sigma_coarse * deltas)
    trans = torch.cumprod(1 - alpha + 1e-10, dim=1)[:, :-1]
    weights_coarse = alpha * torch.cat([torch.ones_like(trans[:, :1]), trans], dim=1)
    # Fine sampling
    z_vals_mid = 0.5 * (z_vals_coarse[:, 1:] + z_vals_coarse[:, :-1])
    z_vals_fine = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1], N_fine, device=device)
    z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_vals_fine], dim=-1), dim=-1)
    pts_fine = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * z_vals_combined.unsqueeze(-1)
    pts_fine_flat = pts_fine.reshape(-1, 3)
    dirs_fine_flat = ray_directions.unsqueeze(1).expand(-1, N_coarse + N_fine, -1).reshape(-1, 3)
    outputs_fine = model(pts_fine_flat, dirs_fine_flat)
    sigma_fine = outputs_fine[:, 0].reshape(N_rays, N_coarse + N_fine)
    rgb_fine = outputs_fine[:, 1:].reshape(N_rays, N_coarse + N_fine, 3)
    # Final rendering
    deltas_fine = z_vals_combined[:, 1:] - z_vals_combined[:, :-1]
    deltas_fine = torch.cat([deltas_fine, delta_inf], dim=1)
    alpha_fine = 1 - torch.exp(-sigma_fine * deltas_fine)
    trans_fine = torch.cumprod(1 - alpha_fine + 1e-10, dim=1)[:, :-1]
    weights_fine = alpha_fine * torch.cat([torch.ones_like(trans_fine[:, :1]), trans_fine], dim=1)
    rgb = torch.sum(weights_fine.unsqueeze(-1) * rgb_fine, dim=1)

    return rgb

def train_nerf(model, training_rays, training_colors, epochs, batch_size, learning_rate, device):
    """
    Trains the NeRF model.
    Params:
    - model: The NeRF model to train
    - training_rays: ray_origins & ray_directions
    - training_colors: Ground truth RGB colors for each ray
    - epochs: Number of training epochs
    - batch_size: Number of rays per batch
    - learning_rate: Learning rate for the optimizer
    - device: Device being used
    Returns:
    - model: Trained NeRF model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rays_o, rays_d = training_rays
    dataset = torch.utils.data.TensorDataset(rays_o, rays_d, training_colors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for rays_o_b, rays_d_b, colors_b in loader:
            rays_o_b = rays_o_b.to(device)
            rays_d_b = rays_d_b.to(device)
            colors_b = colors_b.to(device)
            preds = render_rays(model, rays_o_b, rays_d_b, device = device) # all on CPU
            loss = nn.functional.mse_loss(preds, colors_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * rays_o_b.size(0)

def render_novel_views(model, output_dir, num_frames=30, resolution=(400, 400)):
    """
    Render frames of the scene from new camera viewpoints in a circular trajectory.
    Params:
    - model: Trained NeRF model
    - output_dir: Directory to save rendered frames
    - num_frames: Number of frames to render
    - resolution: Resolution for output images
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    # Renders the frames from different viewpoints along a circular path
    for i in range(num_frames):
        theta = 2 * math.pi * i / num_frames
        camera_pose = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0, 4 * np.cos(theta)],
            [np.sin(theta),  np.cos(theta), 0, 4 * np.sin(theta)],
            [0,             0,             1, 2],
            [0,             0,             0, 1]
        ], dtype=torch.float32)
        rays_o, rays_d = generate_camera_rays(camera_pose, H=resolution[0], W=resolution[1])
        with torch.no_grad():
            rgb = render_rays(model, rays_o, rays_d, near=2.0, far=6.0, device='cpu')
        img = rgb.reshape(resolution[0], resolution[1], 3).numpy()
        imageio.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), (img * 255).astype(np.uint8))


def main_pipeline(data_dir, output_dir):
    """
    NeRF pipeline.
    Params:
    - data_dir: Directory containing the input images and camera poses
    - output_dir: Directory to save output renderings
    """
    device_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/nerf_synthetic"
    if data_dir is None:
        data_dir = "data/nerf_synthetic"
    # Load and prepare image and camera data
    images, poses, camera_angle_x = load_synthetic_images_and_camera_metadata(data_dir)
    N, H, W, _ = images.shape
    if isinstance(camera_angle_x, float):
        # Single class case - use same angle for all images
        camera_angle_x = [camera_angle_x] * N
    elif isinstance(camera_angle_x, list):
        # Multi-class case - ensure we have enough angles
        if len(camera_angle_x) < N:
            camera_angle_x = camera_angle_x * (N // len(camera_angle_x) + 1)
            camera_angle_x = camera_angle_x[:N]
    # Generate rays and collect RGB values for all images
    all_rays_o = []
    all_rays_d = []
    all_rgb = []
    for i in range(N):
        angle_x = camera_angle_x[0] if isinstance(camera_angle_x, list) else camera_angle_x
        rays_o, rays_d = generate_camera_rays(poses[i], H=H, W=W, camera_angle_x=angle_x)
        img = images[i]
        if img.shape[2] > 3: # If RGBA, remove alpha channel
            img = img[..., :3]
        elif img.shape[2] == 1: # If grayscale, convert to RGB
            img = img.expand(-1, -1, 3)
        expected_size = H * W * 3
        if img.numel() != expected_size:
            print(f"Warning: Image {i} has unexpected size {img.shape}, resizing")
            img = img.view(3, H, W).permute(1, 2, 0)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgb.append(img.reshape(H * W, 3))  
    # Combine data from all images
    rays_o = torch.cat(all_rays_o, dim=0) 
    rays_d = torch.cat(all_rays_d, dim=0)
    colors = torch.cat(all_rgb, dim=0)
    # Create the NeRF model
    pos_input_dim = 3
    dir_input_dim = 3
    num_frequencies = 10
    model = NeRFModel(input_pos_dimensions = pos_input_dim,
                      input_dir_dimensions = dir_input_dim,
                      num_frequencies = num_frequencies)
    model.to(device_cpu)
    # Train the NeRF model
    train_nerf(
        model = model, 
        training_rays = (rays_o, rays_d),
        training_colors = colors,
        epochs = 20, 
        batch_size = 1024, 
        learning_rate= 5e-4,
        device = device_cpu)
    render_novel_views(model, "output_frames")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main_pipeline("datadir", "outdir")