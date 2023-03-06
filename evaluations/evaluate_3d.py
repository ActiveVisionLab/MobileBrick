import argparse
import os
import csv
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import open3d as o3d
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import utils.o3d_helper as o3d_helper


def compute_curvature(points, radius=0.005):
    tree = KDTree(points)

    curvature = [ 0 ] * points.shape[0]

    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)
        if len(indices) < 3:
            print("invalid points")
            continue
        # local covariance
        M = np.array([ points[i] for i in indices ]).T
        M = np.cov(M)

        # eigen decomposition
        V, E = np.linalg.eig(M)
        # h3 < h2 < h1
        h1, h2, h3 = V

        curvature[index] = h3 / (h1 + h2 + h3)

    return np.asarray(curvature)


def visibility_test(volume, min_pts, resolution, voxel_size, mesh, device):
    """ filter out points that are not wihin the masked volume

    Args:
        volume (np.ndarray): [H,W,D] the mask volume
        min_pts (np.ndarray): minimum points
        resolution (np.ndarray): volume resolution
        voxel_size (float): voxel_size
        mesh (open3d.mesh): input mesh
        device (string): the device for pytorch
    """

    points = np.asarray(mesh.vertices)
    volume = torch.from_numpy(volume).float().to(device)
    voxels = (points - min_pts) / voxel_size
    voxels = voxels / (resolution-1) * 2 - 1
    voxels = torch.from_numpy(voxels)[..., [2,1,0]].float().to(device)
    mask = F.grid_sample(volume.unsqueeze(0).unsqueeze(0),  # [1,1,H,W,D]
                         voxels.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # [1,1,1,N,3] 
                         mode="nearest",
                         padding_mode="zeros",
                         align_corners=True)  # []
    mask = mask[0, 0, 0, 0].cpu().numpy() > 0
    mesh.remove_vertices_by_mask(mask==False)
    mesh.compute_vertex_normals()
    return mesh


def evaluate(
    pred_points,
    # pred_curv,
    gt_points,
    # gt_curv,
    threshold,
    verbose=False
):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(gt_points)
    distances, indices = nbrs.kneighbors(pred_points)

    pred_gt_dist = np.mean(distances)
    precision = np.sum(distances < threshold) / len(distances)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pred_points)
    distances, indices = nbrs.kneighbors(gt_points)

    # curv_diff = np.abs(gt_curv - pred_curv[indices[:,0]])
    # mean_curv_diff = np.mean(curv_diff)

    gt_pred_dist = np.mean(distances)
    recall = np.sum(distances < threshold) / len(distances)
    F1 = 2 * precision * recall / (precision + recall)
    chamfer = pred_gt_dist + gt_pred_dist

    if verbose:
        # print("pred -> gt: ", pred_gt_dist)
        print("precision @ {}: {:.6f}".format(threshold, precision))
        # print("gt -> pred: ", gt_pred_dist)
        print("recall @ {}: {:.6f}".format(threshold, recall))

        print("F1: {:.6f}".format(F1))
        # print("mean curvature difference: {:.6f}".format(mean_curv_diff))
        print("Chamfer: {:.6f}".format(chamfer))
        # print("{:.3f}/{:.4f}/{:.3f}/{:.4f}/{:.4f}".format(pred_gt_dist, precision, gt_pred_dist, recall, F1))
    out = {}
    out['pred_gt'] = pred_gt_dist
    out['accuracy'] = precision
    out['gt_pred'] = gt_pred_dist
    out['recall'] = recall
    out['chamfer'] = pred_gt_dist + gt_pred_dist
    out['F1'] = F1
    return out


def sample_surface_points(mesh):
    n_points = mesh.vertices


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--seq_txt",
                             default="./evaluations/test_seqs.txt",
                             help="the .txt file listing the testing sequences")
    args_parser.add_argument("--gt_root",
                             default="./data",
                             help="the directory of the dataset")
    args_parser.add_argument("--skip", 
                             nargs="+",
                             help="sequences to skip")
    args_parser.add_argument("--method",
                             required=True,
                             help="name of the method to be evaluated")
    args_parser.add_argument("--n_pts",
                             default=100000,
                             type=int,
                             help="the number of sampling points for evaluation")
    args_parser.add_argument("--save_output",
                             action="store_true",
                             help="whether to save output mesh")
    args = args_parser.parse_args()

    n_samples = args.n_pts
    pred_dir = os.path.join(f"./meshes/{args.method}")
    gt_root = args.gt_root

    skip_seqs = args.skip if args.skip is not None else []
    with open(args.seq_txt, "r") as f:
        seqs = [l for l in f.read().split(",") if l not in skip_seqs]

    chamfer_loss = []
    fitness = []
    accuracy = []
    recall = []
    F1 = []
    accuracy_1,recall_1, F1_1 = [], [], []
    for seq in seqs:
        # load ground-truth
        print(f"evaluating {seq}: ")
        gt_dir = os.path.join(gt_root, seq)
        visibility_mask = np.load(os.path.join(gt_dir, "visibility_mask.npy"), allow_pickle=True).item()
        resolution = visibility_mask['resolutions']
        volume = visibility_mask['mask'].reshape(resolution)
        voxel_size = visibility_mask['voxel_size']
        min_pts = visibility_mask['min_pts']
        gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "mesh", "gt_mesh.ply"))
        gt_points = np.asarray(gt_mesh.sample_points_poisson_disk(n_samples).points)
        # gt_mesh_trimesh = trimesh.load(os.path.join(gt_dir, "mesh", "gt_mesh.ply"))
        # gt_curv = trimesh.curvature.discrete_gaussian_curvature_measure(gt_mesh_trimesh, gt_points, 0.005)

        # load predictions
        mesh_path = os.path.join(pred_dir, f"{seq}.ply")
        pred_mesh_trimesh = trimesh.load(mesh_path)
        pred_mesh = o3d.io.read_triangle_mesh(mesh_path)

        gt_pts = o3d_helper.np2pc(gt_mesh.vertices)
        pred_pts = o3d_helper.np2pc(pred_mesh.vertices)
        threshold = 0.02
        trans_init = np.eye(4)
        reg_p2l = o3d.pipelines.registration.registration_icp(
            gt_pts, pred_pts, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
        fitness.append(reg_p2l.fitness)
        if reg_p2l.fitness > 0.99:
            new_pose = reg_p2l.transformation
            pred_mesh.transform(np.linalg.inv(new_pose))
        pred_mesh = visibility_test(volume, min_pts, resolution, voxel_size, pred_mesh, device)
        if args.save_output:
            o3d.io.write_triangle_mesh(os.path.join(pred_dir, f"{seq}_cropped.ply"), pred_mesh)
        if len(np.asarray(pred_mesh.triangles)) > 0:
            pred_points = np.asarray(pred_mesh.sample_points_poisson_disk(n_samples).points)
            # pred_curv = trimesh.curvature.discrete_gaussian_curvature_measure(pred_mesh_trimesh, pred_points, 0.005)
        else:
            pred_points = np.random.permutation(np.asarray(pred_mesh.vertices))[:n_samples]
        out = evaluate(
            pred_points,
            # pred_curv,
            gt_points,
            # gt_curv,
            threshold=0.0025,
            verbose=True)
        chamfer_loss.append(out['chamfer'])
        accuracy.append(out['accuracy'])
        recall.append(out['recall'])
        F1.append(out['F1'])
        out = evaluate(pred_points, gt_points, threshold=0.005, verbose=True)
        accuracy_1.append(out['accuracy'])
        recall_1.append(out['recall'])
        F1_1.append(out['F1'])

    with open(os.path.join(pred_dir, "data.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(seqs)
        writer.writerow(fitness)
        writer.writerow(chamfer_loss)
        writer.writerow(accuracy)
        writer.writerow(recall)
        writer.writerow(F1)
        writer.writerow(accuracy_1)
        writer.writerow(recall_1)
        writer.writerow(F1_1)
        
    print("final result: ")
    print(f"chamfer: {sum(chamfer_loss) / len(chamfer_loss)}")
    print(f"accuracy: {sum(accuracy) / len(accuracy)}")
    print(f"recall: {sum(recall) / len(recall)}")
    print(f"F1: {sum(F1) / len(F1)}")



if __name__ == "__main__":
    main()

