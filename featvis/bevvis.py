import os
import argparse
import torch
import spconv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.transform import Rotation as R
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch.nn.functional as F

from pcdet.datasets.coda import coda_utils

parser = argparse.ArgumentParser(description='Visualize SparseConvTensor as 2D heatmap')
parser.add_argument('--featdir', type=str, default="../model_feats", help='Path to the directory containing the .pth file containing SparseConvTensor')
parser.add_argument('--featname', type=str, default="spatial_features_2d.pth", help='Layer name for the spconv tensor')
parser.add_argument('--frame', type=int, default=0, help='Start frame to use')
parser.add_argument('--nframes', type=int, default=1, help='Number frames to use')
parser.add_argument('--channel', type=str, default="all", help='Channel to visualize')

def normalize_color(color):
    normalized_color = [(r / 255, g / 255, b / 255) for r, g, b in color]
    return normalized_color

def build_box_lines(boxes_np):
    box_lines_dict = {
        "x": [],
        "y": []
    }
    for box_np in boxes_np:
        x, y, z, l, w, h, heading = box_np

        # Create transformation matrix from the given yaw
        rot_mat = R.from_euler("z", [heading], degrees=False).as_matrix()[:, :2, :2]

        # Half-length, half-width, and half-height
        hl, hw, hh = l / 2.0, w / 2.0, h / 2.0

        # Define 4 corners of the bounding box in the local frame
        local_corners = np.array([
            [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw] # Add fifth point to enclose box
        ]).T
        # Transform corners to the frame
        frame_corners = rot_mat.dot(local_corners)
        frame_corners += np.array([[x], [y]])
        
        # Define lines for the bounding box (each line by two corner indices)
        x_pts = frame_corners[0, 0]
        y_pts = frame_corners[0, 1]
        box_lines_dict["x"].append(x_pts)
        box_lines_dict["y"].append(y_pts)

    return box_lines_dict

def load_bev_features(featdir, featname, frame_offset, channel="all"):
    """
    Accepts feature directory and frame
    Return heatmap [B H W C], voxels, voxelcoords
    """
    bev_featsdir = os.path.join(featdir, "bev")

    feat_name = featname.split('.')[0] + f'_{frame_offset}.pth'
    voxelsname = featname.split('.')[0] + f'_{frame_offset}_voxels.pth'
    voxelcoordsname = featname.split('.')[0] + f'_{frame_offset}_voxel_coords.pth'

    featpath = os.path.join(bev_featsdir, feat_name)
    voxelspath = os.path.join(bev_featsdir, voxelsname)
    voxelcoordspath = os.path.join(bev_featsdir, voxelcoordsname)

    # Input [B C H W] Output [B, H, W, C]
    loaded_spconv_tensor = torch.load(featpath)
    loaded_spconv_tensor = loaded_spconv_tensor.transpose(1, 2).transpose(2, 3)

    # Convert the features to a 2D heatmap
    if channel=="all":
        heatmap = loaded_spconv_tensor
    else:
        heatmap = loaded_spconv_tensor[:, :, :, int(channel)] # check channel dimension 1

    # Load voxel files
    loaded_voxels_tensor = torch.load(voxelspath)
    loaded_voxelcoords_tensor = torch.load(voxelcoordspath)

    return heatmap, loaded_voxels_tensor, loaded_voxelcoords_tensor

def load_bbox_preds(featdir, frame_offset, ax=None):
    predsdir = os.path.join(featdir, "preds")
    predsname = f'predsvis_{frame_offset}.pth'
    predspath       = os.path.join(predsdir, predsname)

    bbox_color_map=normalize_color(coda_utils.BBOX_ID_TO_COLOR)
    if os.path.exists(predspath):
        loaded_preds_tensor = torch.load(predspath)
        loaded_boxes_np = loaded_preds_tensor['pred_boxes'].cpu().detach().numpy()
        loaded_labels_np = loaded_preds_tensor['pred_labels'].cpu().detach().numpy() 

        box_lines_dict = build_box_lines(loaded_boxes_np)
        box_classes = loaded_labels_np

        if ax!=None:
            for box_idx in range(len(box_lines_dict["x"])):
                box_color = bbox_color_map[box_classes[box_idx]]
                ax.plot(-box_lines_dict["y"][box_idx], -box_lines_dict["x"][box_idx], color=box_color)

            return ax

    return box_lines_dict, box_classes

def visualize_sparse_conv_tensor(args):
    featdir, featname, frame, nframes, channel = args.featdir, args.featname, args.frame, \
        args.nframes, args.channel

    # Define a custom colormap with nonlinear mapping
    colors = [(0, 0, 0.5), (0, 0.1, 1), (0, 0.3, 1), (0, 0.5, 1),
          (0.2, 0.7, 1), (0.4, 0.85, 1), (0.6, 0.95, 1), (0.8, 1, 1),
          (0.95, 0.8, 0), (1, 0.7, 0)]  # RGB values
    positions = [i/9 for i in range(10)]  # Evenly spaced positions from 0 to 1
    conv_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    for frame_offset in range(1, nframes):
        # Load BEV features
        heatmap, loaded_voxels_tensor, loaded_voxelcoords_tensor = load_bev_features(featdir, featname, frame_offset)
        heatmap = torch.sum(heatmap, dim=-1) # Reduce channel dimension for direct rgb visualization
        heatmap_min, heatmap_max = torch.min(heatmap), torch.max(heatmap)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        heatmap = (heatmap*255)
        heatmap_np = heatmap.cpu().detach().numpy().astype(np.uint8) # Convert to int for plotting
        loaded_voxels_np = loaded_voxels_tensor.cpu().detach().numpy() # [B X Y Z]
        loaded_voxelcoords_np = loaded_voxelcoords_tensor.cpu().detach().numpy() # [B X Y Z]

        loaded_voxel_points_np = loaded_voxels_np.reshape(-1, 3)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        predsdir = os.path.join(featdir, "preds")
        predsname = f'predsvis_{frame_offset}.pth'
        predspath       = os.path.join(predsdir, predsname)
        # Visualize SparseConvTensor
        ax1.imshow(heatmap_np[0], cmap=conv_cmap)
        ax1.set_title("BEV Features Heatmap")

        # Load bbox predictions
        ax2 = load_bbox_preds(featdir, frame_offset, ax2)

        # Visualize voxel centers
        ax2.scatter(-loaded_voxel_points_np[:, 1], -loaded_voxel_points_np[:, 0], s=1, 
            c=loaded_voxel_points_np[:, 2], cmap='viridis')
        ax2.set_title("Top-Down View of Voxel Grid Centers")

        # Save the plot as an image
        frame_id = frame + frame_offset
        bev_plot_dir = f'{channel}_bev_plots'
        if not os.path.exists(bev_plot_dir):
            os.makedirs(bev_plot_dir)
        output_path = os.path.join(bev_plot_dir, f'{frame_id}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved the plot as {output_path}")

def fit_gp(args):
    featdir, featname, frame = args.featdir, args.featname, args.frame

    #0 Interpolate BEV Feature map to original voxel size
    heatmap, loaded_voxels_tensor, loaded_voxelcoords_tensor = load_bev_features(featdir, featname, frame)
    B, H, W, C = heatmap.shape
    heatmap = heatmap.permute(0, 3, 1, 2)

    original_voxel_size = (1504, 1504)
    scale_factor = (original_voxel_size[0] / H, original_voxel_size[1] / W)
    heatmap_full = F.interpolate(
        input=heatmap, scale_factor=scale_factor, mode='bilinear', align_corners=False
    ).squeeze().permute(1, 2, 0) # H x W x C
    
    #1 Downsample voxel range after interpolation for efficiency
    trunc_voxel_range = (512, 512) # (51.2, 51.2, 6m) with 0.1m voxel res 
    center_h, center_w = original_voxel_size[0] // 2, original_voxel_size[1] // 2
    start_h, end_h = center_h - trunc_voxel_range[0]//2, center_h + trunc_voxel_range[0]//2
    start_w, end_w = center_w - trunc_voxel_range[1]//2, center_w + trunc_voxel_range[1]//2
    heatmap_trunc   = heatmap_full[start_h:end_h, start_w:end_w, :] # HxWxC
    import pdb; pdb.set_trace()

    #2 Plot original point cloud with downsampled voxel size
    loaded_voxels_np = loaded_voxels_tensor.cpu().detach().numpy() # [B X Y Z]
    loaded_voxel_points_np = loaded_voxels_np.reshape(-1, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax2.scatter(-loaded_voxel_points_np[:, 1], -loaded_voxel_points_np[:, 0], s=1, 
        c=loaded_voxel_points_np[:, 2], cmap='viridis')
    ax2.set_title("Top-Down View of Voxel Grid Centers")
    import pdb; pdb.set_trace()

    #1 Manually map feature voxels that correspond to each class to RGB colors
    box_lines_dict, box_classes = load_bbox_preds(featdir, frame_offset)

    #2 Fit GPs for each channel



if __name__ == '__main__':
    args = parser.parse_args()
    fit_gp(args)

    visualize_sparse_conv_tensor(args)