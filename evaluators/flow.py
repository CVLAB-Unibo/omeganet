from __future__ import division
import cv2
import os
import numpy as np
import argparse
import sys

sys.path.insert(0, os.path.abspath(".."))
import helpers.flow_tool.flowlib as fl

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, help="Path to kitti stereo dataset")
parser.add_argument("--prediction_folder", type=str, help="Path to the flow prediction")
args = parser.parse_args()


def main():
    img_num = 200
    noc_epe = np.zeros(img_num, dtype=np.float)
    noc_acc = np.zeros(img_num, dtype=np.float)
    occ_epe = np.zeros(img_num, dtype=np.float)
    occ_acc = np.zeros(img_num, dtype=np.float)

    eval_log = os.path.join(args.prediction_folder, "flow_result.txt")
    with open(eval_log, "w") as el:
        for idx in range(img_num):
            # read groundtruth flow
            gt_noc_fn = args.datapath + "training/flow_noc/%.6d_10.png" % idx
            gt_occ_fn = args.datapath + "training/flow_occ/%.6d_10.png" % idx
            gt_noc_flow = fl.read_flow(gt_noc_fn)
            gt_occ_flow = fl.read_flow(gt_occ_fn)

            # read predicted flow (in png format)
            pred_flow_fn = args.prediction_folder + "%.6d_10.png" % idx
            pred_flow = fl.read_flow(pred_flow_fn)

            # resize pred_flow to the same size as gt_flow
            dst_h = gt_noc_flow.shape[0]
            dst_w = gt_noc_flow.shape[1]

            # evaluation
            (single_noc_epe, single_noc_acc) = fl.evaluate_kitti_flow(
                gt_noc_flow, pred_flow, None
            )
            (single_occ_epe, single_occ_acc) = fl.evaluate_kitti_flow(
                gt_occ_flow, pred_flow, None
            )
            noc_epe[idx] = single_noc_epe
            noc_acc[idx] = single_noc_acc
            occ_epe[idx] = single_occ_epe
            occ_acc[idx] = single_occ_acc
            output_line = (
                "Flow %.6d Noc EPE = %.4f"
                + " Noc ACC = %.4f"
                + " Occ EPE = %.4f"
                + " Occ ACC = %.4f\n"
            )
            el.write(
                output_line
                % (idx, noc_epe[idx], noc_acc[idx], occ_epe[idx], occ_acc[idx])
            )

    noc_mean_epe = np.mean(noc_epe)
    noc_mean_acc =  (1 - np.mean(noc_acc)) * 100.0
    occ_mean_epe = np.mean(occ_epe)
    occ_mean_acc = (1 - np.mean(occ_acc)) * 100.0

    print("Mean Noc EPE = %.2f " % noc_mean_epe)
    print("F1 Noc = %.2f " % noc_mean_acc)
    print("Mean Occ EPE = %.2f " % occ_mean_epe)
    print("F1 Occ = %.2f " % occ_mean_acc)


main()
