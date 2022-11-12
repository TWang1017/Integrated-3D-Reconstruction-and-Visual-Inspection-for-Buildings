import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from PIL import Image
import cv2
from plyfile import PlyData, PlyElement
import open3d as o3d
from sympy import appellf1
from tqdm import tqdm
import sys
from operator import itemgetter
from more_itertools import unique_everseen
from xarray import Coordinate


'''trajectory生成extrinsics文件'''
require_traje = False # 生成了的话就false不需要再生成
if require_traje:
    trajectory = pd.read_csv(r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\scene\trajectory.log", sep=':', header=None)
    save_dir = r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\_Combined\wordCoor.txt"
    srt = 1
    end = 5
    result = []
    trans_num = len(pd.DataFrame(trajectory))

    while end - trans_num != 5:
        '''提取每张照片的transformation matrix（包含rotation还有translation）'''
        '''transformaiton matrix的格式：https://dev.intelrealsense.com/docs/rs-trajectory'''
        trans = pd.DataFrame(trajectory[srt:end])
        trans.columns = ['column']
        trans = trans['column'].str.split(' ', expand=True).astype(float)
        trans = np.array(trans)
        # trans = np.dot(colour2depth, trans)
        trans = np.linalg.inv(trans)
        # print(trans, type(trans))
        result.append(trans)
        srt+=5
        end+=5
    for c, i in enumerate(result):
        mat = np.matrix(i)
        with open(save_dir,'a') as f:
            f.write(f"{c:05d}")
            f.write("\n")
            for n, line in enumerate(mat):
                np.savetxt(f, line)
                # if n == 3:
                #     f.write("\n")



require_pcd = False
if require_pcd:
    vertices = []
    vertex_colors = []
    mask_cam = []
    intrinsics = np.array([
    [633.69268798828125, 0, 640.79150390625],
    [0, 632.9580078125, 363.08261108398438],
    [0,       0,       1]])
    depth_dir = r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\_Combined\depth"
    color_dir = r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\_Combined\color"
    ply_filename = r'C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\_Combined\reconfromProjection\fused.ply'
    down_ply = r'C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\_Combined\reconfromProjection\fused_down.ply'
    scale = 1000
    depth_files = [f for f in os.listdir(depth_dir)]
    color_files = [f for f in os.listdir(color_dir)]
    trajectory = pd.read_csv(r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\scene\trajectory.log", sep=':', header=None)
    srt = 1
    end = 5
    height, width = 720, 1280
    nth = 50 # downsample, fetch data every nth to prevent menmory exploding
        
    for depth, color in tqdm(zip(depth_files[::nth], color_files[::nth]), total= len(color_files[::nth])):
        depth = os.path.join(depth_dir, depth)
        depth = np.array(o3d.io.read_image(depth))
        depth = list(np.concatenate(depth).flat)
        depth = np.array(depth) / scale

        color = os.path.join(color_dir, color)
        color = np.array(o3d.io.read_image(color))
        color = color.reshape(-1, color.shape[-1])   
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x = list(np.concatenate(x).flat)
        y = list(np.concatenate(y).flat)

        trans = pd.DataFrame(trajectory[srt:end])
        trans.columns = ['column']
        trans = trans['column'].str.split(' ', expand=True).astype(float)
        trans = np.array(trans)
        extrinsics = np.linalg.inv(trans)
        srt+=5 * nth
        end+=5 * nth

        xyz_ref = np.matmul(np.linalg.inv(intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertices.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color).astype(np.uint8))

    vertices = np.concatenate(vertices, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)

    vertices = np.array([tuple(v) for v in vertices], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex_all = np.empty(len(vertices), vertices.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertices.dtype.names:
        vertex_all[prop] = vertices[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(ply_filename)
    print("saving the final model to", ply_filename)

    # downsample
    pcd = o3d.io.read_point_cloud(ply_filename)
    # downpcd = pcd.uniform_down_sample(every_k_points=20) # downsample the point cloud by collecting every n-th points.
    downpcd = pcd.voxel_down_sample(voxel_size=0.01) # "Downsample the point cloud with a voxel of 0.02"
    o3d.io.write_point_cloud(down_ply, downpcd)
    print('downsample finished')




require_worldXYZ = True
if require_worldXYZ:
    def color_coordinates(rgb, filename):
        # Load image, ensure not palettised, and make into Numpy array
        pim = Image.open(filename).convert('RGBA')
        data = pim.getdata()
        # print(data[1250])
        im  = np.array(pim)
        # Get X and Y coordinates of all rgb pixels
        Y, X = np.where(np.all(im==rgb,axis=2))
        # 提取指定rgb颜色的所有坐标, 也就是gradcam highlight外围
        coordinates = list(zip(X, Y))
        # print(len(coordinates))

        if len(coordinates)>0:
            # 计算这些坐标的centroid,所有的x平均，所有的y平均就是centroid
            centroid = (sum(X) / len(coordinates), sum(Y) / len(coordinates))
            centroid = int(centroid[0]), int(centroid[1])
            return centroid, coordinates
        else:
            return "specified rgb does not exist"

        # 下面的code可以用来得到指定颜色的pixel的所有index
        # pixel_index = []
        # for i, item in enumerate(data):
        #     if item[0] == rgb[0] and item[1] == rgb[1] and item[2] == rgb[2]:
        #          pixel_index.append(i)
        # print(len(pixel_index))

    color_cam = r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\color_CAM_solid\00116.jpg"
    depth = r"C:\Users\SmartLab\Desktop\traditional recon and CAM\3d_Recon\rosbag\HDB1\depth\00116.png"
    scale = 1000
    height, width = 720, 1280
    centroid, peripheral = color_coordinates((255,255,0,255), color_cam) # check overlay color rgb: https://imagecolorpicker.com/en
    print(centroid)

    points_colour = (0, 0, 0) # centroid颜色
    centroid_colour = (255, 0, 255)
    image = cv2.imread(color_cam)

    for p in peripheral[::305]:
      image = cv2.circle(image, p, radius=0, color=points_colour, thickness=5)
    cv2.imwrite(r"C:\Users\SmartLab\Desktop\final_pointsArray.png", image)
    # 标注centroid
    image = cv2.circle(image, centroid, radius=0, color=centroid_colour, thickness=50) 
    cv2.imwrite(r"C:\Users\SmartLab\Desktop\final_centroid.png", image)

    # 计算centroid的world coordinates
    depth = Image.fromarray(np.array(o3d.io.read_image(depth)))
    depth = depth.getpixel(centroid)
    depth = np.array(depth) / scale
    # print(depth)
    x, y = centroid
    intrinsics = np.array([
    [633.69268798828125, 0, 640.79150390625],
    [0, 632.9580078125, 363.08261108398438],
    [0,       0,       1]])

    extrinsics_00000 = np.array([ # from wordCoor.txt, number 00000
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]])

    extrinsics_00116 = np.array([ # from wordCoor.txt, number 00116.jpg, 
    [9.993919585083016477e-01, 4.064215877158770460e-03, 3.462937977346222662e-02, -3.670549234947621531e-01],
    [-4.213354054330406377e-03, 9.999821522520043748e-01, 4.234979186086842810e-03, 2.775452606529388300e-02],
    [-3.461155021116625596e-02, -4.378310987970284417e-03, 9.993912504202001612e-01, -3.135396216069026576e-02],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]])

    # calculate the coordinates for img 116 with respect to the first image
    # xyz_ref = np.matmul(np.linalg.inv(intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
    # xyz_world = np.matmul(np.linalg.inv(extrinsics_00116), np.vstack((xyz_ref, np.ones_like(x))))[:3]
    # print(xyz_world.transpose((1, 0))[0])

    
    extrinsics_1700 = np.array([ # from wordCoor.txt, number 1700,
    [3.792162022326200921e-01, 9.369110391929643192e-03, -9.252606639355555052e-01, -3.029830160018903484],
    [1.059269922918648493e-01, 9.929353498576339865e-01, 5.346834048910598797e-02, -3.247993932274784168e-01],
    [9.192249698761815191e-01, -1.182861376309481127e-01, 3.755447294026534322e-01, -6.790696621300252644],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]])

    trans_matrix = np.matmul(extrinsics_00000, np.linalg.inv(extrinsics_1700)) # trans_matrix make 1700 to 0000, so 1700 as first img
    print('trans_matrix', trans_matrix)

    extrinsics = np.matmul(extrinsics_00116, trans_matrix) # cal relative extrinsics for 116 with respect to 1700
    print('final_extrinsics', extrinsics)

    # calculate the coordinates for img 116 with respect to the 1700 image
    xyz_ref = np.matmul(np.linalg.inv(intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
    xyz_world = np.matmul(np.linalg.inv(extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
    print(xyz_world.transpose((1, 0))[0])





