"""
Depth evaluation for KITTI Eigen split
This code is based on https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
We would like to thank C. Godard and other authors for sharing their code
"""
from __future__ import division
import os
import argparse
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(".."))
from helpers import depth_utils


parser = argparse.ArgumentParser(description="Evaluation on the KITTI dataset")
parser.add_argument(
    "--prediction_folder", type=str, help="path to estimated disparities", required=True
)
parser.add_argument(
    "--datapath", type=str, help="path to ground truth disparities", required=True
)
parser.add_argument(
    "--min_depth", type=float, help="minimum depth for evaluation", default=1e-3
)
parser.add_argument(
    "--max_depth", type=float, help="maximum depth for evaluation", default=80
)
parser.add_argument(
    "--filename_file",
    type=str,
    help="path to filename file",
    default="../filenames/eigen_test.txt",
)
args = parser.parse_args()

if __name__ == "__main__":
    print("Depth evaluation is started: loading ground-truths and predictions")
    pred_disparities = []
    num_samples = 697

    for t_id in range(num_samples):
        pred_disparities.append(
            np.load(os.path.join(args.prediction_folder, str(t_id) + ".npy"))
        )
    datapath = args.datapath
    if not datapath.endswith("/"):
        datapath += "/"
    test_files = depth_utils.read_text_lines(args.filename_file)
    gt_files, gt_calib, im_sizes, im_files, cams = depth_utils.read_file_data(
        test_files, datapath
    )

    num_test = len(im_files)
    gt_depths = []
    pred_depths = []
    for t_id in range(num_samples):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        depth = depth_utils.generate_depth_map(
            gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True
        )
        gt_depths.append(depth.astype(np.float32))

        disp_pred = pred_disparities[t_id].squeeze()

        # need to convert from disparity to depth
        focal_length, baseline = depth_utils.get_focal_length_baseline(
            gt_calib[t_id], camera_id
        )
        depth_pred = (baseline * focal_length) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0

        pred_depths.append(depth_pred)

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    with tqdm(total=num_samples) as pbar:
        for i in range(num_samples):

            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]
            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

            gt_height, gt_width = gt_depth.shape
            crop = np.array(
                [
                    0.40810811 * gt_height,
                    0.99189189 * gt_height,
                    0.03594771 * gt_width,
                    0.96405229 * gt_width,
                ]
            ).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            # Scale matching
            scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            pred_depth[mask] *= scalor

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            (
                abs_rel[i],
                sq_rel[i],
                rms[i],
                log_rms[i],
                a1[i],
                a2[i],
                a3[i],
            ) = depth_utils.compute_errors(gt_depth[mask], pred_depth[mask])
            pbar.update(1)
    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            "abs_rel", "sq_rel", "rms", "log_rms", "d1_all", "a1", "a2", "a3"
        )
    )
    print(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
            abs_rel.mean(),
            sq_rel.mean(),
            rms.mean(),
            log_rms.mean(),
            d1_all.mean(),
            a1.mean(),
            a2.mean(),
            a3.mean(),
        )
    )
