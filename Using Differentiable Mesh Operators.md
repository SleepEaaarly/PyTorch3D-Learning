# PyTorch3D: Using Differentiable Mesh Operators

## Talking Points

* Examples of differentiable mesh operators
  * Chamfer Distance
* Task 1 : Fitting a mesh to a target 3D shape
* Task 2 : Fitting a mesh to a collection of target 2D images

## Mesh Introduction

### What is a Mesh?

![What is Mesh](D:\CG\pytorch3d学习资料\What is Mesh.png)

### Mesh Operators

![Mesh Operators](D:\CG\pytorch3d学习资料\Mesh Operators.png)

Mesh Operators are functions that take in as argument meshes: f(Mesh)

Differentiability means that we can compute gradients:

df/d(Mesh)

## Example: Chamfer Distance

![Example Chamfer Distance](D:\CG\pytorch3d学习资料\Example Chamfer Distance.png)

#### Implementation of Chamfer Distance

Step 1: Online point sampling from the faces of each mesh

Step 2: L2 distance between two sets of points

#### Step 1: Online point sampling

![Online point sampling](D:\CG\pytorch3d学习资料\Online point sampling.png)

注：(w1, w2, w3)被称为重心坐标(barycentric coordinates)，w1+w2+w3 = 1且非负

## Task 1 : Fitting a mesh to a target 3D shape

![Task1](D:\CG\pytorch3d学习资料\Task1.png)

### Load an obj file

```python
# Load the dolphin mesh.
trg_obj = os.path.join('dolphin.obj')

# We read the target 3D model using load_obj
verts, faces, aux = load_obj(trg_obj)
```

### create a Meshes object
```python
# verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
# faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# We scale normalize and center the target mesh to fit in a sphere of radius centered at (0,0,0)
# Normalizing the target mesh, speeds up the optimization but is not necessary!
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

# Construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])


# We initialize the source shape to be a sphere of radius 1
src_mesh = ico_sphere(4, device)
```

### Optimization loop
```python
# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters if equal to the total number of vertices in src_mesh
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)


# The optimizer
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)


# Defining args
# Number of optimization steps
Niter = 2000
# Weight for the chamfer loss
w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 1.0
# Weight for mesh normal consistency
w_normal = 0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.1
# Plot period for the losses
plot_period = 250
loop = tqdm(range(Niter))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # We sample 5k points from the surface of each mesh
    sample_trg = sample_points_from_meshes(trg_mesh, 5000)
    sample_src = sample_points_from_meshes(new_src_mesh, 5000)
    
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    # Weighted sum of the losses
    loss = ...
    
    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)
    
    # Save the losses for plotting
    chamfer_losses.append(loss_chamfer)
    edge_losses.append(loss_edge)
    normal_losses.append(loss_normal)
    laplacian_losses.append(loss_laplacian)
    
    # Plot mesh
    if i % plot_period == 0:
        plot_pointcloud(new_src_mesh, title="iter: %d" % i)
    
    # Optimization step
    loss.backward()
    optimizer.step()
```

## Task 2 : Fitting a mesh to a collection of 2D images

### Initialization

```python
# initial the source shape to be a sphere of radius 1.
src_mesh = ico_sphere(4, device)

# We arbitrarily choose one particular view that will be used to visualize results
camera = OpenGLPerspectvieCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the -z direction.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
```

### Rasterization & Rendering Setting

```python
# Rasterization settings for differentiable rendering, where the blur_radius initialization is based on Liu et al, 'Soft' Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning', ICCV 2019
sigma = 1e-4
raster_settings_soft = RasterizationSettings(
	image_size=128,
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
    faces_per_pixel=50,		# k value
)

# Differentiable soft renderer using per vertex RGB colors for texture
renderer_textured = MeshRenderer(
	rasterizer=MeshRasterizer(
    	camera=camera,
        raster_settings=raster_settings_soft
    ),
    shader=SoftPhongShader=(
    	device=device,
        cameras=camera,
        lights=lights
    )
)
```

### Optimization setting

```python
# Number of views to optimize over in each SGD iteration 2-view training
number_views_per_iteration = 2
# Number of optimization steps
Niter = 2000
# ...
plot_period = 250

losses = {"rgb": {"weight": 1.0, "values": {}},
       "silhouette": {"weight": 1.0, "values": {}},
         "edge": {"weight": 1.0, "values": {}},
         "normal": {"weight": 0.01, "values": {}},
       "laplacian": {"weight": 1.0, "values": {}},}

verts_shape = src_mesh.verts_packed.shape
deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentrum=0.9)    
```

### Optimization

```python
loop = tqdm(range(Niter))

for i in loop:
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    loss={}
    loss['edge'] = mesh_edge_loss(new_src_mesh)
    loss['normal'] = mesh_normal_consistency(new_src_mesh)
    loss['laplacian'] = mesh_laplacian_smoothing(new_src_mesh, method='uniform')
    
    # Randomly select two views to optimize over in this iteration. Better than one.
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images_predicted = renderer_textured(new_src_mesh, cameras=target_cameras[j], lights=lights)
        
        # silhouette: 影子，轮廓
        predicted_silhouette = images_predicted[..., 3]
        loss_silhouette = ((predicted_silhouette - target_silhouette[j])**2).mean()
        loss["silhouette"] += loss_silhouette / num_views_per_iteration
        
        # RGB
        predicted_rgb = images_predicted[..., :3]
        loss_rgb = ((predicted_rgb - target_rgb[j])**2).mean()
        loss["rgb"] += loss_rgb / num_views_per_iteration
        
    # Weighted sum of the losses
    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        losses[k]["values"].append(l)
        
    loss.set_description("total_loss = %.6f" % sum_loss)
    
    if i % plot_period == 0:
        visualize_prediction(new_src_mesh, renderer=renderer_textured, title="iter: %d" % i, silhouette=False)
    
    # Optimization step
    sum_loss.backward()
    optimizer.step()
```

