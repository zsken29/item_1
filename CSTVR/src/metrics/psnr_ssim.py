import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel


def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算 PSNR (峰值信噪比)。

    参考: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    参数:
        img (ndarray): 范围为 [0, 255] 的图像。
        img2 (ndarray): 范围为 [0, 255] 的图像。
        crop_border (int): 图像边缘裁剪的像素数，不参与计算。
        input_order (str): 输入顺序是 'HWC' 还是 'CHW'。默认值: 'HWC'。
        test_y_channel (bool): 是否在 YCbCr 的 Y 通道上测试。默认值: False。

    返回:
        float: PSNR 结果。
    """

    assert img.shape == img2.shape, (f'图像形状不同: {img.shape}, {img2.shape}。')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'错误的输入顺序 {input_order}。支持的顺序为 "HWC" 和 "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

def _ssim(img, img2):
    """计算单通道图像的 SSIM (结构相似性)。

    由 `calculate_ssim` 函数调用。

    参数:
        img (ndarray): 范围为 [0, 255] 的图像，顺序为 'HWC'。
        img2 (ndarray): 范围为 [0, 255] 的图像，顺序为 'HWC'。

    返回:
        float: SSIM 结果。
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()



def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算 SSIM (结构相似性)。

    参考:
    Image quality assessment: From error visibility to structural similarity

    结果与 https://ece.uwaterloo.ca/~z70wang/research/ssim/ 发布的官方 MATLAB 代码一致。

    对于三通道图像，分别为每个通道计算 SSIM 然后取平均值。

    参数:
        img (ndarray): 范围为 [0, 255] 的图像。
        img2 (ndarray): 范围为 [0, 255] 的图像。
        crop_border (int): 图像边缘裁剪的像素数。
        input_order (str): 输入顺序。
        test_y_channel (bool): 是否在 Y 通道上测试。

    返回:
        float: SSIM 结果。
    """

    assert img.shape == img2.shape, (f'图像形状不同: {img.shape}, {img2.shape}。')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'错误的输入顺序 {input_order}。支持的顺序为 "HWC" 和 "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()
