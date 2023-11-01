import os
import argparse
import torch
import spconv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing
from sklearn.decomposition import PCA
import torch.nn.functional as F

from pcdet.datasets.coda import coda_utils
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

parser = argparse.ArgumentParser(description='Visualize SparseConvTensor as 2D heatmap')
parser.add_argument('--featdir', type=str, default="../model_feats", help='Path to the directory containing the .pth file containing SparseConvTensor')
parser.add_argument('--featname', type=str, default="spatial_features_2d.pth", help='Layer name for the spconv tensor')
parser.add_argument('--frame', type=int, default=1, help='Start frame to use')
parser.add_argument('--nframes', type=int, default=1, help='Number frames to use')
parser.add_argument('--channel', type=str, default="all", help='Channel to visualize')


GRID_SIZE = (1504, 1504)
VOXEL_SIZE = 0.1

SM_GRID_SIZE = (512, 512)
NUM_SAMPLES_PER_CLASS = 1

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

def scale_features(X):
    """
    X - N x D dimensional vector (N samples D dimensions)
    """
    #0 Center each feature dimension to u=0 and std dev=1
    X_scaled = ( X - np.mean(X, axis=-1, keepdims=True) )  / np.std(X, axis=-1, keepdims=True)

    return X_scaled

def read_bbox_file(featdir, frame_offset):
    predsdir = os.path.join(featdir, "preds")
    predsname = f'predsvis_{frame_offset}.pth'
    predspath       = os.path.join(predsdir, predsname)

    if os.path.exists(predspath):
        loaded_preds_tensor = torch.load(predspath)
        loaded_boxes_np = loaded_preds_tensor['pred_boxes'].cpu().detach().numpy()
        loaded_labels_np = loaded_preds_tensor['pred_labels'].cpu().detach().numpy() 

    return loaded_boxes_np, loaded_labels_np

def load_bbox_preds(featdir, frame_offset, ax=None):
    bbox_color_map=normalize_color(coda_utils.BBOX_ID_TO_COLOR)
    loaded_boxes_np, loaded_labels_np = read_bbox_file(featdir, frame_offset)
    
    box_lines_dict = build_box_lines(loaded_boxes_np)
    box_classes = loaded_labels_np

    if ax!=None:
        for box_idx in range(len(box_lines_dict["x"])):
            box_color = bbox_color_map[box_classes[box_idx]]
            ax.plot(-box_lines_dict["y"][box_idx], -box_lines_dict["x"][box_idx], color=box_color)

    return box_lines_dict, box_classes, ax

def predict_color_channel(features, gp):
    """
    Accepts HxWxD scaled features
    Returns HxWx1 labels
    """
    H, W, D = features.shape
    features_flattened = features.reshape(-1, D) 
    pred_flat, std_flat = gp.predict(features_flattened, return_std=True)

    pred, std = pred_flat.reshape(H, W, -1), std_flat.reshape(H, W, -1)
    return pred, std

def get_voxel_centers(voxel_map, voxel_res):
    """
    [HxWxC] voxel map
    [HW x C] XY voxel centers
    """
    H, W, C = voxel_map.shape

    voxel_indices_y = torch.linspace(0, H-1, H).to(device="cuda")
    voxel_indices_x = torch.linspace(0, W-1, W).to(device="cuda")
    rows, cols = torch.meshgrid(voxel_indices_y, voxel_indices_x)
    grid_indices = torch.stack((rows, cols), dim=-1).reshape(-1, 2)
    grid_offset = torch.tensor([(H-1)//2, (W-1)//2], dtype=torch.float32, device="cuda")
    grid_centers =  (grid_indices - grid_offset) * VOXEL_SIZE
    grid_centers = torch.stack((-grid_centers[:, 1], -grid_centers[:, 0]), axis=-1)
    
    # grid_centers = grid * voxel_res + voxel_res / 2

    # # Shift origin to center
    # grid_centers[:, :, 0] -= (H-1)/2 * voxel_res
    # grid_centers[:, :, 1] -= (W-1)/2 * voxel_res

    return grid_centers

def normalize_to_range(values, target_min, target_max):
    values = values.reshape(-1, 3)
    values_min = np.min(values, axis=0, keepdims=True)
    values_max = np.max(values, axis=0, keepdims=True)

    return target_min + (values - values_min) * (target_max - target_min) / (values_max - values_min)

def visualize_sparse_conv_tensor(visualization_dict):
    featdir, featname, frame, nframes, channel, gp_r, gp_g, gp_b = visualization_dict.values()

    # # Define a custom colormap with nonlinear mapping
    colors = [(0, 0, 0.5), (0, 0.1, 1), (0, 0.3, 1), (0, 0.5, 1),
          (0.2, 0.7, 1), (0.4, 0.85, 1), (0.6, 0.95, 1), (0.8, 1, 1),
          (0.95, 0.8, 0), (1, 0.7, 0)]  # RGB values
    positions = [i/9 for i in range(10)]  # Evenly spaced positions from 0 to 1
    conv_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    for frame_offset in range(frame, frame+nframes):
        #0 Load BEV features
        heatmap, loaded_voxels_tensor, loaded_voxelcoords_tensor = load_bev_features(featdir, featname, frame_offset)
        heatmap_full = upscale_heatmap(heatmap)

        #1 Reduce BEV feature map size
        heatmap_full = reduce_heatmap_size(heatmap_full, SM_GRID_SIZE[0], SM_GRID_SIZE[1])

        heatmap_np = heatmap_full.cpu().detach().numpy().astype(np.float32) 
        heatmap_scaled_np = scale_features(heatmap_np.squeeze())

        r_pred, r_std = predict_color_channel(heatmap_scaled_np, gp_r)
        g_pred, g_std = predict_color_channel(heatmap_scaled_np, gp_g)
        b_pred, b_std = predict_color_channel(heatmap_scaled_np, gp_b)
        bev_features_rgb = np.concatenate((r_pred, g_pred, b_pred), axis=-1)
        bev_features_std = np.concatenate((r_std, g_std, b_std), axis=-1)
        
        heatmap_pts = get_voxel_centers(heatmap_full, VOXEL_SIZE) # LiDAR frame
        heatmap_pts_np = heatmap_pts.cpu().detach().numpy()
        bev_features_rgb = np.clip(bev_features_rgb.reshape(-1, 3), 0, 1)
        bev_features_std = normalize_to_range(bev_features_std, 0, 1)
        loaded_voxels_np = loaded_voxels_tensor.cpu().detach().numpy() # [B X Y Z]
        loaded_voxelcoords_np = loaded_voxelcoords_tensor.cpu().detach().numpy() # [B X Y Z]

        loaded_voxel_points_np = loaded_voxels_np.reshape(-1, 3)
  
        # Limit range of voxel points
        loaded_voxel_points_mask = np.logical_and(
            np.logical_and(loaded_voxel_points_np[:, 0]>=-25.6, loaded_voxel_points_np[:,0]<=25.6),
            np.logical_and(loaded_voxel_points_np[:, 1]>=-25.6, loaded_voxel_points_np[:,1]<=25.6),
        )
        loaded_voxel_points_np = loaded_voxel_points_np[loaded_voxel_points_mask]

        #2 Plot BEV conv features with bounding boxes
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        predsdir = os.path.join(featdir, "preds")
        predsname = f'predsvis_{frame_offset}.pth'
        predspath       = os.path.join(predsdir, predsname)
        ax1.scatter(heatmap_pts_np[:, 1], heatmap_pts_np[:, 0], s=1,
            c=bev_features_rgb)
        _, _, ax1 = load_bbox_preds(featdir, frame_offset, ax1)
        ax1.set_title("BEV Conv Features")

        #3 Plot point cloud with bounding boxes
        _, _, ax2 = load_bbox_preds(featdir, frame_offset, ax2)
        ax2.scatter(-loaded_voxel_points_np[:, 1], -loaded_voxel_points_np[:, 0], s=1, 
            c=loaded_voxel_points_np[:, 2], cmap='viridis')
        ax2.set_title("3D Point Cloud")

        # #4 Plot uncertainties for RGB values
        # _, _, ax3 = load_bbox_preds(featdir, frame_offset, ax3)
        # cax3 = ax3.scatter(heatmap_pts_np[:, 1], heatmap_pts_np[:, 0], s=1, 
        #     c=bev_features_std, cmap='viridis')
        
        # # fig.colorbar(cax3, ax=ax3)
        # std_cmap = LinearSegmentedColormap.from_list("custom", bev_features_std)
        # # Extract scalars from the RGB values (for the colorbar)
        # scalars = np.linspace(0, 1, 100)

        # # Create a mappable object based on the custom colormap
        # mappable = cm.ScalarMappable(cmap=std_cmap)
        # mappable.set_array(scalars)
        # cbar = fig.colorbar(mappable, ax=ax3)
        # cbar.set_label('Std dev values')
        # ax3.set_title("BEV Conv Feature Uncertainty")

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

def get_pts_in_boxes(pc, featdir, frame):
    loaded_boxes_np, loaded_labels_np = read_bbox_file(featdir, frame)

    #0 Load point clouds, bounding boxes, and labels
    loaded_boxes    = torch.from_numpy(loaded_boxes_np).to(device="cuda").unsqueeze(0)  # [B N 7]
    loaded_pc       = torch.from_numpy(pc).to(device="cuda").unsqueeze(0)               # [B N 3]

    #1 Gets associated box for each point
    box_idxs_of_pts = points_in_boxes_gpu(loaded_pc, loaded_boxes).reshape(-1,)
    box_idxs_of_pts = box_idxs_of_pts.cpu().detach().numpy().astype(int)

    valid_box_idxs_of_pts = box_idxs_of_pts[box_idxs_of_pts!=-1]
    box_idx_to_label    = loaded_labels_np[valid_box_idxs_of_pts]
    
    #2 Gets points in boxes and their corresponding colors
    in_box_pts      = loaded_pc[0, box_idxs_of_pts!=-1, :].cpu().detach().numpy() # Points in boxes

    return in_box_pts, box_idx_to_label

def upscale_heatmap(heatmap, new_H=GRID_SIZE[0], new_W=GRID_SIZE[1]):
    """
    Input Pytorch [1 x H x W x C]
    Output Pytorch [H x W x C]
    """
    B, H, W, C = heatmap.shape
    heatmap = heatmap.permute(0, 3, 1, 2)

    scale_factor = (new_H / H, new_W / W)
    heatmap_full = F.interpolate(
        input=heatmap, scale_factor=scale_factor, mode='bilinear', align_corners=False
    ).squeeze().permute(1, 2, 0) # H x W x C

    return heatmap_full

def reduce_heatmap_size(heatmap_full, sm_H, sm_W):
    """
    Input Pytorch [H x W x C]
    Output Pytorch [H x W x C]
    """
    H, W, C = heatmap_full.shape
    center_y, center_x = H // 2, W // 2

    top = max(0, center_y - sm_H // 2)
    left = max(0, center_x - sm_W // 2)
    
    return heatmap_full[top:top+sm_H, left:left+sm_W, :]

def fit_gp(args):
    """
    heatmap                 -
    loaded_voxels_tensor    - in the  
    """
    featdir, featname, frame = args.featdir, args.featname, args.frame

    #0 Interpolate BEV Feature map to original voxel size
    heatmap, loaded_voxels_tensor, loaded_voxelcoords_tensor = load_bev_features(featdir, featname, frame)
    heatmap_full = upscale_heatmap(heatmap)

    #2 Extract xyz points and colors for bounding boxes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    loaded_voxels_np = loaded_voxels_tensor.cpu().detach().numpy() # [B x 3] LiDAR Frame
    loaded_voxel_points_np = loaded_voxels_np.reshape(-1, 3)
    in_box_pts_lidar, in_box_labels = get_pts_in_boxes(loaded_voxel_points_np, featdir, frame)
    bbox_color_map      = np.array(normalize_color(coda_utils.BBOX_ID_TO_COLOR))
    in_box_colors   = bbox_color_map[in_box_labels] # Maps points to colors

    # Map from lidar to image pixel coordinate frame
    in_box_pts_image = np.stack((in_box_pts_lidar[:, 1], in_box_pts_lidar[:, 0], in_box_pts_lidar[:, 2]), axis=-1)
    in_box_pts_image[:, :2] = -in_box_pts_image[:, :2]

    # For each class get a single box pt and color
    box_set = {}
    sampled_box_pts_image     = np.empty((0, 3), dtype=np.float32)
    sampled_box_pts_lidar     = np.empty((0, 3), dtype=np.float32)
    sampled_box_colors  = np.empty((0, 3), dtype=np.float32)
    for box_idx, box_label in enumerate(in_box_labels):
        if box_label not in box_set:
            box_set[box_label] = 0

        if box_set[box_label]<NUM_SAMPLES_PER_CLASS: # Sample at most 100 points from each box
            box_pt_image = in_box_pts_image[box_idx].reshape(-1, 3)
            box_pt_lidar = in_box_pts_lidar[box_idx].reshape(-1, 3)
            box_color = bbox_color_map[box_label].reshape(-1, 3)
            sampled_box_pts_image = np.concatenate((sampled_box_pts_image, box_pt_image))
            sampled_box_pts_lidar = np.concatenate((sampled_box_pts_lidar, box_pt_lidar))
            sampled_box_colors = np.concatenate((sampled_box_colors, box_color))
            box_set[box_label] += 1

    ax1.scatter(in_box_pts_image[:, 0], in_box_pts_image[:, 1], s=1, c=in_box_colors)
    ax1.set_title("Top-Down View of Points in Bboxes by Color")

    #3 Plot original point cloud with bounding boxes in image frame
    remap_loaded_voxel_points_np = np.stack((loaded_voxel_points_np[:, 1], loaded_voxel_points_np[:, 0], loaded_voxel_points_np[:, 2]), axis=-1)
    remap_loaded_voxel_points_np[:, :2] = -remap_loaded_voxel_points_np[:, :2]

    ax2.scatter(remap_loaded_voxel_points_np[:, 0], remap_loaded_voxel_points_np[:, 1], s=1, 
        c=remap_loaded_voxel_points_np[:, 2], cmap='viridis')
    _, _, ax2 = load_bbox_preds(featdir, frame, ax2)
    ax2.set_title("Top-Down View of Voxel Grid Centers")

    #4 Map pts to BEV feature voxels ()
    offset = np.array([GRID_SIZE[0] // 2, GRID_SIZE[1] // 2]) * VOXEL_SIZE
    grid_indices = np.floor( (sampled_box_pts_image[:, :2] + offset) / VOXEL_SIZE).astype(int)

    grid_indices[:, 0] = np.clip(grid_indices[:, 0], 0, GRID_SIZE[0]-1)
    grid_indices[:, 1] = np.clip(grid_indices[:, 1], 0, GRID_SIZE[1]-1) # correct indices as box
    grid_indices_torch = torch.from_numpy(grid_indices).to(device="cuda", dtype=torch.long)
    grid_features = heatmap_full[grid_indices_torch[:, 0], grid_indices_torch[:, 1], :]

    ax3.scatter(grid_indices[:, 0], grid_indices[:, 1], s=1, c=sampled_box_colors)
    ax3.set_title("BEV Feature Centers")

    plt.savefig("gpfit.png")
    #5 Sample training points to fit gaussian processes
    np.random.seed(42)
    NUM_SAMPLES = min(100, grid_features.shape[0])
    selected_indices_1d = np.random.choice(grid_features.shape[0], NUM_SAMPLES, replace=False)

    grid_features_np = grid_features.cpu().detach().numpy()
    selected_grid_features_np   = grid_features_np[selected_indices_1d, :]
    selected_color_values       = sampled_box_colors[selected_indices_1d, :]
    
    #6 Fit GPs for each channel    
    X_scaled = scale_features(selected_grid_features_np)
    kernel = RBF(1, (1e-2, 1e2))
    num_restarts = int(1e2)
    gp_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=num_restarts).fit(
        X_scaled, selected_color_values[:, 0]
    )
    gp_g = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=num_restarts).fit(
        X_scaled, selected_color_values[:, 1]
    )
    gp_b = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=num_restarts).fit(
        X_scaled, selected_color_values[:, 2]
    )

    #3 Fit GPs for each channel
    return gp_r, gp_g, gp_b


if __name__ == '__main__':
    args = parser.parse_args()
    gp_r, gp_g, gp_b = fit_gp(args)

    visualization_dict = vars(args)
    visualization_dict["gp_r"] = gp_r
    visualization_dict["gp_g"] = gp_g
    visualization_dict["gp_b"] = gp_b
    # visualization_dict = {arg:  for arg in args}
    visualize_sparse_conv_tensor(visualization_dict)