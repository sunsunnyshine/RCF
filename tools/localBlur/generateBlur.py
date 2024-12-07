import numpy as np
import torch
from tools.localBlur.blur import blur_image


def generateBlurFlow(gt_fw_flows, gt_bw_flows, fw_flow_agg, fw_residual_adjustment, bw_flow_agg,
                     bw_residual_adjustment, img, gt, lookup_table):
    '''
    Args:
        gt_fw_flows (list): list of forward ground truth flows
        gt_bw_flows (list): list of backward ground truth flows
        fw_flow_agg (list): list of forward flow aggregation
        fw_residual_adjustment (list): list of forward residual adjustment
        bw_flow_agg (list): list of backward flow aggregation
        bw_residual_adjustment (list): list of backward residual adjustment
        gts (list): list of ground truth images
    '''
    fw_flow = (gt_fw_flows - fw_flow_agg) * (1 - gt) + fw_residual_adjustment * gt
    bw_flow = (gt_bw_flows - bw_flow_agg) * (1 - gt) + bw_residual_adjustment * gt
    return blur_image(img, fw_flow[0], fw_flow[1], bw_flow[0], bw_flow[1], lookup_table, mask=None,
                      flow_range=16, visualization=False)
