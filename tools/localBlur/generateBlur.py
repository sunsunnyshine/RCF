import torch
import torch.nn.functional as F
from tools.localBlur.blurtest import process_large_image


def generateBlurFlow(gt_fw_flows, gt_bw_flows, fw_flow_agg, fw_residual_adjustment, bw_flow_agg,
                     bw_residual_adjustment, img, gt, origin_img, lookup_table, flow_range, title_size):
    origin_H, origin_W = origin_img.shape[-2:]

    ## step1: 统一到网络输入尺寸
    resized_origin_img = F.interpolate(origin_img, size=img.shape[-2:], mode='bilinear')
    gt_fw_flows = F.interpolate(gt_fw_flows[:, 0], size=img.shape[-2:], mode='bilinear')
    gt_bw_flows = F.interpolate(gt_bw_flows[:, 0], size=img.shape[-2:], mode='bilinear')
    fw_residual_adjustment = F.interpolate(fw_residual_adjustment, size=img.shape[-2:], mode='bilinear')
    bw_residual_adjustment = F.interpolate(bw_residual_adjustment, size=img.shape[-2:], mode='bilinear')
    gt = F.interpolate(gt.unsqueeze(0).float(), size=img.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

    ## step2： 生成模糊图像
    # 背景模糊，前景清晰
    fw_flow_sharp = torch.mul(gt_fw_flows - fw_flow_agg[..., None, None],
                              (1 - gt[:, None, ...]))
    bw_flow_sharp = torch.mul(gt_bw_flows - bw_flow_agg[..., None, None],
                              (1 - gt[:, None, ...])) + torch.mul(bw_residual_adjustment, gt[:, None, ...])
    # 背景模糊，前景局部模糊
    fw_flow_blur = fw_flow_sharp + torch.mul(fw_residual_adjustment, gt[:, None, ...])
    bw_flow_blur = bw_flow_sharp + torch.mul(bw_residual_adjustment, gt[:, None, ...])

    # input: image[B,3,H,W], flow_forward[B,2,H,W], flow_backward[B,2,H,W], lookup_table[B,ker*ker,H,W], flow_range, tile_size
    blur_image_objectsharp = process_large_image(resized_origin_img, fw_flow_sharp, bw_flow_sharp, lookup_table,
                                                 flow_range, title_size)
    # output: numpy [B,C,H,W]
    blur_image_objectblur = process_large_image(resized_origin_img, fw_flow_blur, bw_flow_blur, lookup_table,
                                                flow_range, title_size)

    ## step3: 统一到原图尺寸
    blur_image_objectsharp = F.interpolate(blur_image_objectsharp, size=(origin_H, origin_W), mode='bilinear')
    blur_image_objectblur = F.interpolate(blur_image_objectblur, size=(origin_H, origin_W), mode='bilinear')

    return blur_image_objectsharp, blur_image_objectblur
