import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def get_kernel_from_lookup_table(lookup_table, flow, flow_range=16):
    flow = torch.clamp(flow, -flow_range, flow_range)
    grid = (flow.to(lookup_table.dtype) / flow_range)
    sampled_kernels = F.grid_sample(lookup_table, torch.flip(grid, dims=[-1]), mode='bilinear', align_corners=True)
    # return:(kernel*kernel,H,W)
    return sampled_kernels.squeeze()


def prepare_kernels(u_forward, v_forward, lookup_table, flow_range=16):
    u_forward = u_forward.permute(0, 2, 3, 1)
    v_forward = v_forward.permute(0, 2, 3, 1)
    _, M, N, _ = u_forward.shape
    device = u_forward.device
    K_f = get_kernel_from_lookup_table(lookup_table, u_forward, flow_range)
    K = K_f
    del K_f
    K_b = get_kernel_from_lookup_table(lookup_table, v_forward, flow_range)
    K = 0.5 * K + 0.5 * K_b
    del K_b
    kernels = K.permute(1, 2, 0).view(M, N, 2 * flow_range + 1, 2 * flow_range + 1)
    # return (H,W,kernel,kernel)
    return kernels


def blur_image_with_unfold(image, flow_forward, flow_backward, lookup_table, flow_range=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        kernels = prepare_kernels(flow_forward, flow_backward, lookup_table, flow_range)

        blurred_image = torch.zeros((image.shape[0], image.shape[1], flow_forward.shape[-2], flow_forward.shape[-1]),
                                    device=device, dtype=torch.float32)
        for c in range(image.size(1)):
            # since the unfold only padding 0, we need to crop more area to achieve the same size
            patches = F.unfold(image[:, c:c + 1, :, :], kernel_size=(2 * flow_range + 1, 2 * flow_range + 1))
            patches = patches.permute(0, 2, 1).contiguous().view(flow_forward.shape[-2], flow_forward.shape[-1],
                                                                 2 * flow_range + 1, 2 * flow_range + 1)

            blurred_patches = patches * kernels
            blurred_image[:, c:c + 1, :, :] = blurred_patches.sum(dim=(-2, -1))
            del patches, blurred_patches
        del image, kernels
        torch.cuda.empty_cache()

    return blurred_image


def process_large_image(image, flow_forward, flow_backward, lookup_table, flow_range=16, tile_size=150):
    H, W = image.shape[2], image.shape[3]
    blurred_image = torch.zeros(image.shape, device=image.device, dtype=torch.float32)

    # Pad the image with kernel size (tensor)
    image = F.pad(image.to(torch.float32), (flow_range, flow_range, flow_range, flow_range), mode='reflect')

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = image[:, :, y:y + 2 * flow_range + tile_size, x:x + 2 * flow_range + tile_size]
            tile_blurred = blur_image_with_unfold(tile, flow_forward[..., y:y + tile_size, x:x + tile_size],
                                                  flow_backward[..., y:y + tile_size, x:x + tile_size], lookup_table,
                                                  flow_range)
            blurred_image[..., y:y + tile_size, x:x + tile_size] = tile_blurred

    # return tensor (B,C,H,W)
    return blurred_image


if __name__ == "__main__":
    image = cv2.imread("test/00001.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_numpy = image
    flow_forward = (torch.load("test/fw_residual_adjustment.pth")).squeeze().permute(1, 2, 0).numpy()
    flow_backward = (torch.load("test/bw_residual_adjustment.pth")).squeeze().permute(1, 2, 0).numpy()
    mask = (torch.load("test/mask.pth"))[0, 3, ...].detach().cpu().numpy()

    flow_forward = cv2.resize(flow_forward, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    flow_backward = cv2.resize(flow_backward, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    tau = 1.0
    num_frame = 20
    flow_range = 50
    kernel_size = flow_range * 2 + 1
    tile_size = 200
    lookup_table_file = f'lookup_table_{kernel_size}_{flow_range}_{tau}_{num_frame}.npy'

    if os.path.exists(lookup_table_file):
        lookup_table = np.load(lookup_table_file)
        lookup_table = torch.tensor(lookup_table, dtype=torch.float32).cuda()
        lookup_table = torch.flatten(lookup_table, start_dim=-2, end_dim=-1).unsqueeze(0).permute(0, 3, 1, 2)

    # image: ndarray(480,854,3), flow_forward: ndarray(480,854,2), flow_backward: ndarray(480,854,2), lookup_table: tensor(1, 10201, 101,101)
    # to_tensor
    flow_forward = torch.tensor(flow_forward).permute(2, 0, 1).unsqueeze(0).cuda()
    flow_backward = torch.tensor(flow_backward).permute(2, 0, 1).unsqueeze(0).cuda()
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
    # input: image[B,3,H,W], flow_forward[B,2,H,W], flow_backward[B,2,H,W], lookup_table[B,ker*ker,H,W], flow_range, tile_size
    blurred_image = process_large_image(image, flow_forward, flow_backward, lookup_table, flow_range, tile_size)
    # output: tensor (B,C,H,W)

    # visualize the first image
    blurred_image = blurred_image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    plt.figure(figsize=(10, 5))
    plt.title('Original Image')
    plt.imshow(image_numpy)
    plt.title('Blurred Image')
    plt.imshow(blurred_image)
    plt.show()

    # save
    plt.imshow(blurred_image)
    plt.savefig("blurred_image_fast.png")
