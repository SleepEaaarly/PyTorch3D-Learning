# Making 3D Prediction with 2D Supervision

## Key Challenge for 3D Recognition: Supervision

* 2D supervision: Category label, bounding box, segmentation.
  Can be annotated by non-expert humans.
* 3D supervision: 3D shape, depth, camera pose, etc.
  Not easily annotated by people!

**Claim: ** For 3D Recognition to succeed, we must rely primarily on 2D supervision.

## Task

* Task: Supervised Shape Prediction
* Tool: Differentiable Rendering + PyTorch3D
* Task: Unsupervised Shape Prediction
* Task: Single-Image View Synthesis

### Mesh R-CNN

![Mesh R-CNN](.\pictures\Mesh R-CNN.png)

**How to use only 2D supervision for 3D task?**

![Idea Render and Compare](.\pictures\Idea Render and Compare.png)

### Differentiable Rendering

#### Traditional Rendering

* Rasterization：**Not** differentiable
* Shading: differentiable

### Unsupervised Shape Prediction

![Unsupervised Shape Prediction](.\pictures\Unsupervised Shape Prediction.png)

### Single Image View Synthesis

![Single Image View Synthesis](.\pictures\Single Image View Synthesis.png)

#### **Challenge**

* Need to know depth
* Inpainting Missing Regions

#### Goals

* Complex, realistic scenes
* Train w/o GT Depth; only (Image, RT, Image)
* Test with a single image; (Image, RT) -> Image
* End-To-End training

#### Approach

![Single mesh View Synthesis Approach](.\pictures\Single mesh View Synthesis Approach.png)