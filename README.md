# Integrated-3D-Reconstruction-and-Visual-Inspection-for-Buildings

1. train the model under transfer learning framework
2. use the model the identify defects and visualise using Grad-CAM
3. manipulate the camera intrinsic and extrinsic parameters to quantitatively calculate the defect location
4. you can reconstruct you model either use colmap or open3d instead or even python (see codes in worldCor) to combine the 3D and 2D information for qualitative assessment

Note: input data could either be the paired RGB and depth maps, or ROSBAG file if you choose OPEN3D
