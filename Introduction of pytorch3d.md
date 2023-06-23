# Introduction to PyTorch3D 

本文源自youtube **PyTorch3D @ SIGGRAPH ASIA 2020 Course**。

## Outline

* Motivation
* Goals
* Core Components
* Rendering
* Examples

## Motivation

### Training a DL model

* 2D example: 图片分类总体流程

![2dexample](D:\CG\pytorch3d学习资料\pictures\2dexample.png)

* 3D dl遇到的问题：
  * Batch分组tensor维度不固定
  * 图形学操作(如渲染)不支持梯度

![2dvs3d](D:\CG\pytorch3d学习资料\pictures\2dvs3d.png)

## Goals

深度学习库特性(快速、模块化、可微) + 3D库特性(3D、不统一batching、结合3D操作)

![goals](D:\CG\pytorch3d学习资料\pictures\goals.png)

## Components

![image-20230620111822144](C:\Users\Mars\AppData\Roaming\Typora\typora-user-images\image-20230620111822144.png)

* Data **structures**：batching logic以及使得操作损失函数和渲染过程高效支持不统一维度batching。


### Meshes

#### Representation

![meshes](D:\CG\pytorch3d学习资料\pictures\meshes.png)

三种表示形式：

* list
* Packed Tensor
* Padded Tensor

#### Definition

```python
import torch
from pytorch3d import Meshes

verts_list = [torch.tensor([...]), ...,torch.tensor([...])]
faces_list = [torch.tensor([...]), ...,torch.tensor([...])]

mesh_batch = Meshes(verts=verts_list, faces=faces_list)
```

#### Representation transformation

```python
# packed representation
verts_packed = mesh_batch.verts_packed()

#auxillary tensors
mesh_to_vert_idx = mesh_batch.mesh_to_verts_packed_first_idx()
vert_to_mesh_idx = mesh_batch.verts_packed_to_mesh_idx()

# edges
edges = mesh_batch.edges_packed()

# face normals
face_normals = mesh_batch.faces_normals_packed()
```

### IO && Transforms

#### Load obj

```python
import torch
from pytorch3d import load_obj

verts, faces, aux = load_obj(obj_file)

faces = faces.verts_idx
normals = aux.normals
textures = aux.verts_uvs
materials = aux.material_colors
tex_maps = aux.texture_images
```

#### Load objs as meshes

```
import torch
from pytorch3d import load_objs_as_meshes

batched_mesh = load_objs_as_meshes([obj_file1, obj_file2, obj_file3])
```

#### Composable 3D Transforms

```python
import torch
from pytorch3d.transforms import Transform3d, Rotate, Translate

# example 1
T = Translate(torch.FloatTensor([[1.0, 2.0, 3.0]]))
R = Rotate(torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
RT = Transform3d().compose(R, T)

# example 2
T = Transform3d().scale(2, 1, 3).translate(1, 2, 3)
```

### Ops

#### K Nearest Neighbors

![K Nearest Neighbor](D:\CG\pytorch3d学习资料\pictures\K Nearest Neighbor.png)

对给定点p，寻找点集Q中最近的K个点

 ```python
 import torch
 from pytorch3d.ops import knn_points
 
 N, P1, P2, D, K = 32, 128, 256, 3, 1
 pts1 = torch.randn(N, P1, D)
 pts2 = torch.randn(N, P2, D)
 dists, idx, knn = knn_points(pts1, pts2, K=K)
 ```

#### Graph Conv

![Graph Conv](D:\CG\pytorch3d学习资料\pictures\Graph Conv.png)

对每个Mesh的每个顶点，都有一个特征向量feature，图卷积操作是相邻点之间特征向量的求和平均过程。

```python
import torch
from pytorch3d.ops import GraphConv

conv = GraphConv(input_dim, output_dim, init="normal", directed=False)

# given a mesh which is a Meshes object
verts = mesh.verts_packed()
edges = mesh.edges_packed()
y = conv(verts, edges)
```

### Loss function

#### Chamfer Loss

**Chamfer distance** 是集合S1中每个点对集合S2中点的KNN求和并反之亦然。

![Chamfer Loss](D:\CG\pytorch3d学习资料\pictures\Chamfer Loss.png)

```python
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

sphere_mesh1 = ico_sphere(level=3)
sphere_mesh2 = ico_sphere(level=1)

sample_sphere = sample_points_from_meshes(sphere_mesh1, 5000)
sample_test = sample_points_from_meshes(sphere_mesh2, 5000)
loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)
```

## Rendering

### Rendering: why?

**rendering**：3d模型 -> 2d图片

"在渲染过程中添加梯度使得此步骤可微并应用于DL"是待探索的领域。

### Rendering = rasterization + shading

#### Rasterization

1. 三角面片在2d平面的相交情况
2. z-buffering：比较深度并渲染

均为离散步骤，不可微

#### Shading

给像素增加环境属性，如**光照**、**纹理**

#### Differentiable Rendering

在训练中进行可微渲染的涵义？

场景设置：mesh、texture map、lights、camera

![rendering scene](D:\CG\pytorch3d学习资料\pictures\differentiable rendering.png)

* 正向传播：生成图片并计算loss
* 反向传播：更新场景属性参数

#### 传统光栅化不可微原因

##### Problem 1

![rasterization process 1](D:\CG\pytorch3d学习资料\pictures\rasterization process 1.png)

问题：**Z discontinuity**当triangle mesh在z轴位移微元dz时，输出像素颜色会产生突变(先后顺序突变)。

解决方案：**soft aggregation**考虑z轴方向最近的K个三角形mesh，每个mesh都对像素颜色有贡献。

##### Problem 2

![rasterization process 2](D:\CG\pytorch3d学习资料\pictures\rasterization process 2.png)

问题：**XY discontinuity**当triangle mesh在xy平面位移微元dx(dy)时，输出像素颜色会产生突变(像素是否属于某三角形突变)。

解决方案：**soft aggregation**考虑增加模糊边界。

### PyTorch3D Rendering Engine

* Separate rasterizer & shader modules
* 2 step rasterization
* Return Top K faces in Fragments
* Heterogeneous batching
* Shading in PyTorch

![Rendering engine](D:\CG\pytorch3d学习资料\pictures\Rendering engine.png)

```python
from pytorch3d.renderer import (
	OpenGLPerspectiveCameras, look_at_view_transform, 
    RasterizationSettings, BlendParams, 
    MeshRenderer, MeshRasterizer, HardPhongShader
)
R, T = look_at_view_transform(2.7, 10, 20)
# dist, elev, azim
'''
dist – distance of the camera from the object
elev – angle in degrees or radians. This is the angle between the vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
azim – angle in degrees or radians. The vector from the object to the camera is projected onto a horizontal plane y = 0. azim is the angle between the projected vector and a reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
'''
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(
	image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,	# sets the value of K
)

renderer = MeshRenderer(
	rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras)
)
image = renderer(mesh)

# compute a loss given a ground truth image
loss = (gt_image - image).sum()
loss.backward()
```

#### Blending

different blending modes

#### Mesh texturing options

* Vertex textures
* Texture Map + Vertex UV coordinates
* Texture Atlas

![Mesh texturing options](D:\CG\pytorch3d学习资料\pictures\Mesh texturing options.png)



