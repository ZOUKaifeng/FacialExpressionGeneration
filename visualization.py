import open3d as o3d
from PIL import Image, ImageDraw
import os
import numpy as np
from tqdm import tqdm
import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.mesh import Textures
# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def img_to_gif(image, path):
    images = []
    for img in image:
        img = Image.fromarray(img, 'RGB')
        images.append(img)
    images[0].save(path+"/expression.gif", save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

def set_render():
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    R, T = look_at_view_transform(2.7, 0, 0) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    return renderer

if __name__ == '__main__':
    renderer = set_render()

    data = {}
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    data_path = "../data/BU_FACE/"
    BU_path = "/media/kzou/LaCie/FACE3D_DB/BU/Preprocessed/"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    subject_list = os.listdir(BU_path)
    for sub in subject_list:
        subject_path = os.path.join(BU_path, sub)
        if sub == "LM_indices.txt":
            continue
        subjects = os.listdir(subject_path)
        

        for s in tqdm(subjects):
            data[s] = {}
            exp_path = os.path.join(subject_path, s)
            exp_lists = os.listdir(exp_path)
            
            save_sub_path = os.path.join(data_path, s)
            if not os.path.exists(save_sub_path):
                os.mkdir(save_sub_path)
            

            print(exp_path)
            for exp in exp_lists:
                data[s][exp] = {}
                frames_path = os.path.join(exp_path, exp)
                frame =  os.listdir(frames_path)
                LM = np.zeros((100, 68, 3))
                vert = []
            
            
                save_exp_path = os.path.join(save_sub_path, exp)
                if not os.path.exists(save_exp_path):
                    os.mkdir(save_exp_path) 

                count = 0
                imgs = []
                for f in frame:
                    f_ = f.split(".")
                    if len(f_)<2:
                        continue
                    if f_[1] == "_mesh":
                        count += 1
                        num = int(f_[0])
                        if num != count-1:
                            print(frames_path)
                        if count > 100:
                            break
                      
                        verts, faces, _ = load_obj(os.path.join(frames_path, f))
                        faces = faces.verts_idx
                        texture = loadmat(os.path.join(frames_path, f_[0]+'.mat'))
                        verts_tex = torch.tensor(texture["face_texture"]/255)[None]
                        textures = Textures(verts_rgb=verts_tex).to(device)
                        test_mesh = Meshes([verts], [faces], textures).cuda()
                        images = renderer(test_mesh)
                        images[images<0] = 0
                        imgs.append(images[0, ..., :3].cpu().numpy())
                print(save_exp_path)
                img_to_gif(imgs, save_exp_path)
