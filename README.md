# **NeRF** PyTorch
### **NeRF**(Neural Radiance Fields) re-implementation with minimal code and maximal readability using PyTorch.

You can find original paper in [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.](https://arxiv.org/abs/2003.08934) 

- - -

## Objective

Even though there are lots of NeRF implementation, including [official implementation](https://github.com/bmild/nerf), 
which is written in Tensorflow, and [pytorch implementation](https://github.com/yenchenlin/nerf-pytorch), the codes of that projects are quite hard to read and understand.
And it takes a lot of time and effort to fully digest the entire code. So our main purpose of this project is to **re-implement** the **official NeRF code** following 
three design goals. 

- Minimal Possible Code

- Maximize Readability

- Closely follow the overall structure of [original implementation](https://github.com/bmild/nerf), such that understanding this code will eventually help understanding the original code.

## Data

- We will use the data from official NeRF data repository.

- Go to the following [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and download ```nerf_synthetic``` folder. 
  - As I mentioned, our purpose is to accomodate readers to easily digest the code and thus to understand what is NeRF. So we just use ```nerf_synthetic``` data and this is enough to understand NeRF. 
Handling various kind of data will disturb our main objective. However, if there is some requests for handling other kinds of data, please let me know.

- Place the ```nerf_synthetic``` folder inside the main project directory.

## Result

Before proceeding, we present final rendered images from out project.

<img src="./image_README/final_gif/chair.gif" height="250" width="250"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="./image_README/final_gif/ship.gif" height="250" width="250"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img src="./image_README/final_gif/mic.gif" height="250" width="250">

Images are rendered at 30 degrees latitude.

If you want to see the results of other object, go to ```./image_README/final_gif/``` folder.
