import torch
import numpy as np
import open3d as o3d

from patrik.dataloader import SFDataset4D
from vis.utils import visualize_flow3d


def rel_poses2traj(rel_poses):
    pose = torch.eye(4, dtype=rel_poses.dtype, device=rel_poses.device).unsqueeze(0)
    poses = pose.clone()
    for i in range(len(rel_poses)):
        pose = pose @ torch.linalg.inv(rel_poses[i])
        poses = torch.cat([poses, pose], dim=0)
    return poses


def main():
    dataset = SFDataset4D(dataset_type='waymo', data_split='*', n_frames=1)  # can preload more frames
    frame = 80

    data = dataset[frame]  # iteration should work too

    # Visualize the first point cloud which is stored in the data['pc1'] variable
    pc1 = np.array(data['pc1'])
    flow = np.array(data['gt_flow'])[..., :3]
    id_mask = np.array(data['id_mask1'])

    # Reshape the point cloud to be a list of points
    pc1 = pc1.reshape(-1, 3)
    flow = flow.reshape(-1, 3)
    id_mask = id_mask.reshape(-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc1)
    pcd.normals = o3d.utility.Vector3dVector(flow)

    # Set point colors based on the ID mask
    colors = np.zeros_like(pc1, dtype=np.float64)
    unique_ids = np.unique(id_mask)
    for unique_id in unique_ids:
        mask = (id_mask == unique_id)
        colors[mask] = np.random.rand(3)  # Assign a random color for each ID

    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


class Visualizer(object):
    def __init__(self):
        self.ds = SFDataset4D(dataset_type='waymo', data_split='*', n_frames=1)
        self.vis = o3d.visualization.Visualizer()
        self.key_to_callback = {}
        self.key_to_callback[ord("N")] = self.next_frame
        self.key_to_callback[ord("B")] = self.prev_frame
        self.key_to_callback[ord("Q")] = self.quit
        self.key_to_callback[ord("K")] = self.change_background
        self.key_to_callback[ord("S")] = self.disable_shading

        # Press "H" for help
        # Press "L" to disable lighting

        self.frame = 0
        self.pcd = o3d.geometry.PointCloud()

    def get_point_cloud(self, frame: int, id: int = None):
        data = self.ds[frame]
        pc1 = np.array(data['pc1'])
        flow = np.array(data['gt_flow'])[..., :3]
        id_mask = np.array(data['id_mask1'])

        # Reshape the point cloud to be a list of points
        pc1 = pc1.reshape(-1, 3)
        flow = flow.reshape(-1, 3)

        self.pcd.points = o3d.utility.Vector3dVector(pc1)
        self.pcd.normals = o3d.utility.Vector3dVector(flow)

        if id is not None:
            id_mask = id_mask.reshape(-1)
            colors = np.zeros_like(pc1, dtype=np.float64)
            id_indices = np.where(id_mask == id)[0]

            # Colorize the points with the given ID with red
            colors[id_indices] = np.array([1, 0, 0])
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def next_frame(self, vis):
        print(f"Next frame: {self.frame}")
        self.find_next_occurrence(14)
        self.get_point_cloud(self.frame, id=14)
        vis.update_geometry(self.pcd)
        vis.update_renderer()
        return False

    def prev_frame(self, vis):
        self.find_prev_occurrence(14)
        self.get_point_cloud(self.frame, id=14)
        vis.update_geometry(self.pcd)
        vis.update_renderer()
        return False

    @staticmethod
    def quit(vis):
        vis.destroy_window()
        return True

    @staticmethod
    def change_background(vis):
        opt = vis.get_render_option()
        if np.asarray(opt.background_color).any() != 0:
            opt.background_color = np.asarray([0, 0, 0])
        else:
            opt.background_color = np.asarray([1, 1, 1])
        return False

    @staticmethod
    def disable_shading(vis):
        opt = vis.get_render_option()
        opt.light_on = False
        return False

    def find_next_occurrence(self, id: int):
        while self.frame < len(self.ds):
            self.frame += 1
            data = self.ds[self.frame]
            id_mask = np.array(data['id_mask1'])
            if id in id_mask:
                break

    def find_prev_occurrence(self, id: int):
        while self.frame >= 0:
            self.frame -= 1
            data = self.ds[self.frame]
            id_mask = np.array(data['id_mask1'])
            if id in id_mask:
                break

    def visualize_instance(self, id: int):
        self.find_next_occurrence(id)
        self.get_point_cloud(self.frame)
        o3d.visualization.draw_geometries_with_key_callbacks([self.pcd], self.key_to_callback)


def main2():
    dataset = SFDataset4D(dataset_type='waymo', data_split='*', n_frames=1)  # can preload more frames
    frame = 80

    data = dataset[frame]  # iteration should work too

    # Visualize the first point cloud which is stored in the data['pc1'] variable
    pc1 = np.array(data['pc1'])
    flow = np.array(data['gt_flow'])[..., :3]
    id_mask = np.array(data['id_mask1'])

    # Reshape the point cloud to be a list of points
    pc1 = pc1.reshape(-1, 3)
    flow = flow.reshape(-1, 3)
    id_mask = id_mask.reshape(-1)

    # Get available ids
    unique_ids = np.unique(id_mask)
    print(unique_ids)

    # Filter out points with id 0
    mask = (id_mask == 14)
    pc1 = pc1[mask]
    flow = flow[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc1)
    pcd.normals = o3d.utility.Vector3dVector(flow)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def visualize_flow_demo():
    from ops.filters import filter_grid

    ds = SFDataset4D(dataset_type='waymo', n_frames=40)
    frame_1 = 0
    frame_2 = 1

    pc1 = ds[frame_1]['pc1']
    pc2 = ds[frame_2]['pc1']
    flow = ds[frame_1]['gt_flow']

    visualize_flow3d(pc1, pc2, flow)


def global_cloud_demo():
    from vis.open3d import visualize_points3D
    from matplotlib import pyplot as plt
    from ops.filters import filter_grid

    ds = SFDataset4D(dataset_type='waymo', n_frames=40)
    # i = np.random.randint(len(ds))
    i = 0
    poses12 = ds[i]['relative_pose']
    # construct path from relative poses
    poses = rel_poses2traj(poses12)

    # generate global cloud
    clouds = ds[i]['pc1']
    global_cloud = clouds[0]
    for i in range(1, len(clouds)):
        cloud = clouds[i] @ poses[i][:3, :3].T + poses[i][:3, 3][None]
        global_cloud = torch.cat([global_cloud, cloud], dim=0)
    global_cloud = filter_grid(global_cloud, grid_res=0.5)
    visualize_points3D(global_cloud)

    plt.figure()
    plt.plot(poses[:, 0, 3], poses[:, 1, 3], 'o-')
    plt.axis('equal')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.visualize_instance(14)
