# Volumetric Reconstruction in Pytorch3D

## Outline

* Rendering implicit surfaces in PyTorch3D
* Rendering volumes in PyTorch3D
* Learning 3D Object Categories from Videos in the Wild

## Rendering Implicit surface

### Implicit surface function

![Implicit surface](.\pictures\Implicit_surface.png)

### Rendering an implicit surface

![Rendering an implicit surface](.\pictures\Rendering_an implicit_surface.png)

```python
### PyTorch3D implementation:
implicit_renderer = pytorch3d.renderer.ImplicitRenderer(
	raysampler = ..., raymarcher = ...,
)
f = torch.nn.Module(...) # implicit surface
image = implicit_renderer(implicit_function=f, cameras=camera) # rendering
```

### Raysampling

![Raysampling](.\pictures\Raysampling.png)

```python
raysampler = pytorch3d.renderer.NDCGridRaysampler(
	image_width: int,
    image_height: int,
    n_pts_per_ray: int,
    min_depth: float,
    max_depth: float,
)
raysampler = pytorch3d.renderer.MonteCarloRaysampler(
	n_pts_per_ray: int,
    min_depth: float,
    max_depth: float,
)
```

### Raymarching

![Raymarching](.\pictures\Raymarching.png)

```python
### PyTorch3D implementation:
raymarcher = pytorch3d.renderer.EmissionAbsorbtionRaymarcher()
```

### Nerf

#### overall structure

![Nerf](.\pictures\Nerf.png)

#### 1 Initialize the implicit renderer

```python
# Initialize the raysampler
raysampler_mc = MonteCarloRaysampler(
	min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=3.0,
)

# Initialize the ray marcher
raymarcher = EmissionAbsorptionRaymarcher()

# Create renderer with ray sampler and ray marcher
renderer_mc = pytorch3d.renderer.ImplicitRenderer(
	raysampler=raysampler_mc, raymarcher=raymarcher,
)
```

#### 2a Define the neural radiance field

```python
class NeuralRadianceField(torch.nn.Module):
    def __init__(self, input_dim=60):
        super().__init__()
        self.mlp = torch.nn.Sequential(
        	torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        
        self.occupancy_layer = torch.nn.Sequential(
        	torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )
        
        self.color_layer = torch.nn.Sequential(
        	torch.nn.Linear(256+input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
            torch.nn.Sigmoid(),
        )
```

#### 2b Define NeRF forward pass

```python
def forward(self, ray_bundle: RayBundle):
    # We first convert the ray_bundle to world points
    rays_points_world = ray_bundle_to_ray_points(ray_bundle)
    # rays_points_world.shape = [minibatch * ... * 3]
    
    # self.mlp maps each harmonic embedding to a latent feature space
    features = self.mlp(harmonic_embedding(rays_points_world))
    # features.shape = [minibatch * ... * 256]
    
    # execute the density and color branches
    rays_densities = self.density_layer(features)
    # rays_densities.shape = [minibatch * ... * 1]
    rays_colors = self.color_layer(
    	torch.cat((features, ray_bundle.directions), dim=-1)
    ) # rays_colors.shape = [minibatch * ... * 3]
    
    return rays_densities, rays_colors
```

#### 3 Run optimization loop

```python
# Create NeRF object
neural_radiance_field = NeuralRadianceField()
# Init the optimizer object
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=1e-3)

# The main optimization loop
for iteration in range(10000):
    # Sample random batch index
    batch_idx = torch.randint(len(cameras))
    # Sample a random camera
    camera = training_cameras[batch_idx]
    
    optimizer.zero_grad()
    
    # Run the Monte Carlo renderer
    renderer_pixels, sampled_rays = renderer_mc(
    	cameras = camera,
        volumetric_function = neural_radiance_field
    )
    
    # Sample ground truth images at ray locations
    ground_truth_pixels = sample_images_at_mc_locations(
    	ground_truth_images[batch_idx],
        sampled_rays.xys
    )
    
    # Compute the color error
    loss = (rendered_pixels - ground_truth_pixels).abs().mean()
    
    # Take the optimization step
    loss.backward()
    optimizer.step()
```

## Volume rendering in PyTorch3D

### Voxel grids

![Voxel grids](.\pictures\Voxel_grids.png)

### Voxel grids in PyTorch3D

```python
### PyTorch3D Volumes data structure
volumes = Volumes(
	densities: torch.Tensor,	# batch * density_dim * D * H * W
    features: torch.Tensor,		# batch * feature_dim * D * H * W
)
```

**Also supports volumes of heterogenous size**

### Volume rendering

![Volume rendering](.\pictures\Volume_rendering.png)

```python
### PyTorch3D implementation:
volume_renderer = pytorch3d.renderer.VolumeRenderer(
	raysampler = ..., raymarcher = ...,
)
volumes = pytorch3d.structures.Volumes(...) # Volumes object
iamge = volume_renderer(volumes=volumes, cameras=camera) # rendering
```

### Conversion of point clouds to volumes

![Conversion of point clouds to volumes](.\pictures\Conversion_of point_clouds_to_volumes.png)

```python
pointclouds = Pointclouds(...)
initial_volumes = Volumes(...)

volumes = pytorch3d.ops.add_pointclouds_to_volumes(
	pointclouds,
	initial_volumes,
	mode = "trilinear"	# trilinear | nearest
)
```

