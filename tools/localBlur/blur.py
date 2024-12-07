import numpy as np
from scipy.sparse import coo_matrix
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
import os
import torch
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_optical_flow(image, flow):
    hsv = np.zeros_like(image)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_rgb

def compute_flow_influence(u, v, tau, num_frame, M, N):
    n_elems = (4 * num_frame + 1)
    i = np.zeros(n_elems, dtype=int)
    j = np.zeros(n_elems, dtype=int)
    k = np.zeros(n_elems, dtype=float)

    one_over_num_frame = 1.0 / (num_frame + 1)
    tau_f_over_num_frame = np.array([tau * f / num_frame for f in range(num_frame + 1)])

    count = 0
    # center point
    center_idx = (M // 2) * N + (N // 2)

    tmp_idx_row = center_idx + 1
    tmp_idx_col = center_idx + 1
    i[count] = tmp_idx_row
    j[count] = tmp_idx_col
    k[count] = one_over_num_frame
    # tmp_dist = np.sqrt(u ** 2 + v ** 2)
    # tmp_coeff_of_zero = 1.0 / (1 + tmp_dist)
    # k[count] = tmp_coeff_of_zero
    count += 1

    x = N // 2
    y = M // 2
    for f in range(1, num_frame + 1):
        dx = u * tau_f_over_num_frame[f]
        dy = v * tau_f_over_num_frame[f]

        x1, y1 = int(np.floor(x - dx)), int(np.floor(y - dy))
        x2, y2 = int(np.ceil(x - dx)), int(np.ceil(y - dy))

        x1 = max(0, min(N - 1, x1))
        y1 = max(0, min(M - 1, y1))
        x2 = max(0, min(N - 1, x2))
        y2 = max(0, min(M - 1, y2))

        wx = (x - dx) - x1
        wy = (y - dy) - y1

        w11 = (1 - wx) * (1 - wy)
        w12 = (1 - wx) * wy
        w21 = wx * (1 - wy)
        w22 = wx * wy

        tmp_idx_col_11 = y1 * N + x1 + 1
        tmp_idx_col_12 = y2 * N + x1 + 1
        tmp_idx_col_21 = y1 * N + x2 + 1
        tmp_idx_col_22 = y2 * N + x2 + 1

        i[count] = tmp_idx_row
        j[count] = tmp_idx_col_11
        k[count] = one_over_num_frame * w11
        count += 1

        i[count] = tmp_idx_row
        j[count] = tmp_idx_col_12
        k[count] = one_over_num_frame * w12
        count += 1

        i[count] = tmp_idx_row
        j[count] = tmp_idx_col_21
        k[count] = one_over_num_frame * w21
        count += 1

        i[count] = tmp_idx_row
        j[count] = tmp_idx_col_22
        k[count] = one_over_num_frame * w22
        count += 1

    return i, j, k


def create_lookup_table(kernel_size=33, flow_range=16, tau=1.0, num_frame=70):
    lookup_table = np.zeros((2 * flow_range + 1, 2 * flow_range + 1, kernel_size, kernel_size))
    for i, u in enumerate(range(-flow_range, flow_range + 1)):
        for j, v in enumerate(range(-flow_range, flow_range + 1)):
            i_vals, j_vals, k_vals = compute_flow_influence(u, v, tau, num_frame, kernel_size, kernel_size)
            K = coo_matrix((k_vals, (i_vals - 1, j_vals - 1)),
                           shape=(kernel_size * kernel_size, kernel_size * kernel_size))
            center_idx = (kernel_size // 2) * kernel_size + (kernel_size // 2)
            lookup_table[i, j, ...] = K.toarray()[center_idx].reshape((kernel_size, kernel_size))
    return lookup_table


def get_kernel_from_lookup_table(lookup_table, u, v, flow_range=16):
    u_idx = np.clip(u + flow_range, 0, 2 * flow_range)
    v_idx = np.clip(v + flow_range, 0, 2 * flow_range)
    interpolator = RegularGridInterpolator(
        (np.arange(2 * flow_range + 1), np.arange(2 * flow_range + 1)),
        lookup_table,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    return interpolator((u_idx, v_idx))


def mx_make_kernel_from_flows(look_up_table, flow_range, u_forward, v_forward, u_backward, v_backward):
    k_forward = get_kernel_from_lookup_table(look_up_table, u_forward, v_forward, flow_range)
    k_backward = get_kernel_from_lookup_table(look_up_table, u_backward, v_backward, flow_range)

    # Merge forward and backward kernels
    K = 0.5 * k_forward + 0.5 * k_backward

    return K


def blur_image(image, u_forward, v_forward, u_backward, v_backward, lookup_table, mask=None,
               flow_range=16,visualization=False):
    M, N, _ = image.shape
    blurred_image = np.zeros_like(image)
    pad_size = flow_range  # Half of the kernel size (33 // 2)
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    threshold = 0.7
    for y in range(0, M):
        for x in range(0, N):
            if mask is not None and mask[y, x] < threshold:
                blurred_image[y, x, :] = image[y, x, :]
                continue
            u_f = u_forward[y, x]
            v_f = v_forward[y, x]
            u_b = u_backward[y, x]
            v_b = v_backward[y, x]
            K = mx_make_kernel_from_flows(lookup_table, flow_range, u_f, v_f, u_b, v_b)
            # 可视化center kernel
            if visualization and y == 50 and x == 50:
                plt.imshow(K, cmap='gray')
                plt.show()
            region = padded_image[y:y + 2 * flow_range + 1, x:x + 2 * flow_range + 1]
            blurred_image[y, x, :] = np.sum(region * K[:, :, np.newaxis], axis=(0, 1))

    return blurred_image


# Example usage
if __name__ == "__main__":

    fast_test = False
    crop = False
    visualize_flow = True
    # image = (torch.load("test/img.pth")).permute(1, 2, 0).numpy()
    image = cv2.imread("test/00001.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flow_forward = (torch.load("test/fw_residual_adjustment.pth")).squeeze().permute(1, 2, 0).numpy()
    flow_backward = (torch.load("test/bw_residual_adjustment.pth")).squeeze().permute(1, 2, 0).numpy()
    mask = (torch.load("test/mask.pth"))[0, 3, ...].detach().cpu().numpy()

    # since the flow fields are not in the same size as the image, we need to resize them
    flow_forward = cv2.resize(flow_forward, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    flow_backward = cv2.resize(flow_backward, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    u_forward = flow_forward[..., 0]
    v_forward = flow_forward[..., 1]
    u_backward = flow_backward[..., 0]
    v_backward = flow_backward[..., 1]
    if visualize_flow:
        flow_rgb_forward = visualize_optical_flow(image, flow_forward)
        flow_rgb_backward = visualize_optical_flow(image, flow_backward)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Forward Optical Flow')
        plt.imshow(flow_rgb_forward)
        plt.subplot(1, 2, 2)
        plt.title('Backward Optical Flow')
        plt.imshow(flow_rgb_backward)
        plt.show()


    if fast_test:
        image = cv2.imread("data/data_davis/JPEGImages/480p/camel/00001.jpg")
        flow_forward = np.load("data/data_davis/Flows_NewCT/480p/camel/00001.npy")
        flow_backward = np.load("data/data_davis/BackwardFlows_NewCT/480p/camel/00001.npy")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop:
        # crop a smaller area of the image and flow fields
        image_rgb = image
        crop_size = 100  # define the size of the cropped area
        start_x = image_rgb.shape[1] // 2 - crop_size // 2
        start_y = image_rgb.shape[0] // 2 - crop_size // 2

        image_cropped = image_rgb[start_y:start_y + crop_size, start_x:start_x + crop_size]
        flow_forward_cropped = flow_forward[start_y:start_y + crop_size, start_x:start_x + crop_size]
        flow_backward_cropped = flow_backward[start_y:start_y + crop_size, start_x:start_x + crop_size]
        mask = mask[start_y:start_y + crop_size, start_x:start_x + crop_size]

        u_forward = flow_forward_cropped[..., 0]
        v_forward = flow_forward_cropped[..., 1]
        u_backward = flow_backward_cropped[..., 0]
        v_backward = flow_backward_cropped[..., 1]
        image = image_cropped

    tau = 1.0
    num_frame = 20
    flow_range = 50
    kernel_size = flow_range * 2 + 1

    lookup_table_file = f'lookup_table_{kernel_size}_{flow_range}_{tau}_{num_frame}.npy'

    if os.path.exists(lookup_table_file):
        lookup_table = np.load(lookup_table_file)
    else:
        lookup_table = create_lookup_table(kernel_size=kernel_size, flow_range=flow_range, tau=tau, num_frame=num_frame)
        np.save(lookup_table_file, lookup_table)

    blurred_image = blur_image(image, u_forward, v_forward, u_backward, v_backward, lookup_table, mask,
                               flow_range)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('Blurred Image')
    plt.imshow(blurred_image)
    plt.show()

    plt.imshow(blurred_image)
    plt.savefig("blurred_image.png")
