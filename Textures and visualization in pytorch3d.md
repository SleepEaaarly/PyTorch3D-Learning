# Textures and visualization in pytorch3D

## view the mesh

### mesh and texture

```python
plt.figure(figsize=(5,5))
plt.imshow(images[0,...,:3].cpu().numpy())
# 只传入前三个通道rgb，不传入alpha
```

![view the mesh](.\pictures\view the mesh.png)

### plotly view the mesh only topology interactively

```
plot_batch_individually(cow_mesh)
```

![topology interactively](.\pictures\topology interactively.png)

## understand what we have

![Understand what we have](.\pictures\Understand what we have.png)

## Image stored in texture

### only map

```python
plt.figure(figsize=(7,7))
texture_image = cow_mesh.textures.maps_padded()
plt.imshow(texture_image.squeeze().cpu().numpy())
plt.axis("off")
```

![texture map](.\pictures\texture map.png)

### with vertices points on it

```python
plt.figure(figsize=(7,7))
texturesuv_image_matplotlib(cow_mesh.textures, subsample=None)
plt.axis("off")
```

![texture map with points](.\pictures\texture map with points.png)

## Some mesh manipulations

两头牛

```python
offset1 = cow_mesh.verts_padded().new_tensor([0,0.5,-0.5]).expand(cow_mesh.verts_packed().shape)
offset2 = cow_mesh.verts_padded().new_tensor([0,-0.5,0]).expand(cow_mesh.verts_packed().shape)
small_mesh = cow_mesh.scale_verts(0.4).offset_verts_(offset1)
double_mesh = join_meshes_as_scene([cow_mesh.offset_verts(offset2), small_mesh])
two_meshes = join_meshes_as_batch([cow_mesh, double_mesh])
```

## Closer look at what renderer does

### Rasterizing

```python
fragments = renderer.rasterizer(cow_mesh)
```

### Shading

```
texels = cow_mesh.textures.sample_textures(fragments, faces_packed=cow_mesh.faces_packed())
colors = phong_shading(cow_mesh,fragments,lights,cameras,Materials(device=device),texels)
```

## TexturesVertex

![TexturesVertex](.\pictures\TexturesVertex.png)

## Sampling a pointcloud from a mesh

```python
points, features = sample_points_from_meshes(cow_mesh, num_samples=20000, return_textures=True)
pointcloud = Pointclouds(points=points, features=features)
plot_batch_individually(pointcloud, pointcloud_max_points=19999)
```

![pointcloud](.\pictures\pointcloud.png)
