import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
from os import path as osp
import numpy as np
import cv2
import yaml
from utils.model_utils import get_model_total_params
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

def test_vid4(data_dir, base_out_p, weight_base_p, test_dataset_name):
    """ 在 Vid4 数据集上进行测试。"""
    from arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
    from data.vid4_dataset import Vid4
    from torchvision import transforms
    from arch.IMSM import IND_inv3D
    from utils.options import yaml_load

    device = torch.device('cuda')
    Time_factor = 2
    Scale_factor = 1

    # 根据配置选择重缩放模型架构
    if (Time_factor == 2 and Scale_factor == 1):
        from arch.Mynet_arch import RescalerNet
    else:
        from arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet

    train_dataset_name = 'vimeo'
    out_p = os.path.join(base_out_p, f'Tx{Time_factor}_Sx{Scale_factor}', test_dataset_name)
    weight_p = os.path.join(weight_base_p, f'Tx{Time_factor}_Sx{Scale_factor}_{train_dataset_name}')

    # 加载可逆网络模型 (Inverter)
    inv_root_p = os.path.join(weight_p, 'inverter/config.yml')
    inv_opt = yaml_load(inv_root_p)['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight_p = os.path.join(weight_p, 'inverter/model.pth')
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'], strict=True)

    # 加载重缩放模型 (Rescaler)
    model_root_path = os.path.join(weight_p, 'rescaler/config.yml')
    rescale_opt = yaml_load(model_root_path)
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to(device)
    rescale_weight_p = os.path.join(weight_p, 'rescaler/model.pth')
    weight = torch.load(rescale_weight_p)
    rescale_model.load_state_dict(weight['params'], strict=True)
    rescale_model.eval()
    param = get_model_total_params(rescale_model)
    print(f'参数量: {param}')

    # 确定下采样后的时间维度
    if rescale_opt['down_type']['T'] == 1:
        down_t = 7
    else:
        down_t = 4

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vid4(data_dir, transform, gop_size=7)
    psnr_list = []
    ssim_list = []
    psnr_list_y = []
    ssim_list_y = []

    for ix, data in enumerate(dataset):
        this_scene, GT = data
        imgs = GT.unsqueeze(0)
        raw_h = imgs.shape[-2]
        raw_w = imgs.shape[-1]
        # 填充图像大小以适应网络要求的 32 倍数
        new_h, new_w = raw_h + (32 - raw_h % 32), raw_w + (32 - raw_w % 32)
        imgs_new = torch.zeros(1, 3, 7, new_h, new_w).cuda()
        imgs_new[:, :, :, :raw_h, :raw_w] = imgs
        imgs = imgs_new

        img_p = this_scene[0].split('/')[-2]
        this_p = osp.join(out_p, img_p)

        if not os.path.exists(this_p):
            os.makedirs(this_p)

        with torch.no_grad():
            b, c, t, h, w = imgs.shape
            down_shape = (down_t, h // rescale_opt['down_type']['S'], w // rescale_opt['down_type']['S'])
            T, H, W = imgs.shape[2:]
            # 简单的分块处理演示 (1x2 分块)
            y_grid = 1
            x_grid = 2
            y_size = H // y_grid
            x_size = W // x_grid
            y_d_size = down_shape[1] // y_grid
            x_d_size = down_shape[2] // x_grid
            B = 1
            place_holder_la = torch.zeros(B, 3, down_shape[0], down_shape[1], down_shape[2])
            place_holder_back = torch.zeros(B, 3, down_shape[0], down_shape[1], down_shape[2])
            place_holder_quan = torch.zeros(B, 3, down_shape[0], down_shape[1], down_shape[2])
            place_holder = torch.zeros(B, 3, T, H, W)

            for i in range(y_grid):
                for j in range(x_grid):
                    patch = [i * y_size, (i + 1) * y_size, j * x_size, (j + 1) * x_size]
                    patch_d = [i * y_d_size, (i + 1) * y_d_size, j * x_d_size, (j + 1) * x_d_size]
                    img_pa = imgs[:, :, :, patch[0]:patch[1], patch[2]:patch[3]].to(device)
                    down_size_p = (down_shape[0], down_shape[1] // y_grid, down_shape[2] // x_grid)

                    # 下采样
                    x_down = rescale_model.inference_down(img_pa, down_size_p)
                    # 潜在特征转 RGB
                    LR_img = model.inference_latent2RGB(x_down)

                    LR_img = LR_img.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0
                    LR_img = LR_img.astype(np.uint8)

                    LR_img_ten = torch.from_numpy(LR_img).unsqueeze(0).permute(0, 4, 1, 2, 3).to(device) / 255.0
                    
                    # 逆向重建
                    rev_back = model.inference_RGB2latent(LR_img_ten)
                    # 上采样还原
                    out = rescale_model.inference_up(rev_back, (t, h // y_grid, w // x_grid))

                    # 合并结果
                    place_holder_la[:, :, :, patch_d[0]:patch_d[1], patch_d[2]:patch_d[3]] = x_down.detach().cpu()
                    place_holder_back[:, :, :, patch_d[0]:patch_d[1], patch_d[2]:patch_d[3]] = rev_back.detach().cpu()
                    place_holder_quan[:, :, :, patch_d[0]:patch_d[1], patch_d[2]:patch_d[3]] = LR_img_ten.detach().cpu()
                    place_holder[:, :, :, patch[0]:patch[1], patch[2]:patch[3]] = out.detach().cpu()

            # 最终结果后处理
            LR_img = place_holder_quan.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0
            x_down = place_holder_la.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0
            rev_back = place_holder_back.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0
            out = place_holder.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0
            out = out[:, :raw_h, :raw_w, :] # 裁剪回原始尺寸

            # 创建子目录
            quan_p = os.path.join(this_p, 'quan')
            latent_p = os.path.join(this_p, 'latent')
            rev_p = os.path.join(this_p, 'rev')
            sr_p = os.path.join(this_p, 'sr')
            os.makedirs(quan_p, exist_ok=True)
            os.makedirs(latent_p, exist_ok=True)
            os.makedirs(rev_p, exist_ok=True)
            os.makedirs(sr_p, exist_ok=True)

            # 保存图像
            if down_t == 4:
                for i in range(down_t):
                    index = this_scene[2 * i].split('/')[-1].split('.')[0]
                    cv2.imwrite(os.path.join(quan_p, f'{index}_quan.png'), LR_img[i][:, :, ::-1])
                    cv2.imwrite(os.path.join(latent_p, f'{index}_latent.png'), x_down[i][:, :, ::-1])
                    cv2.imwrite(os.path.join(rev_p, f'{index}_back.png'), rev_back[i][:, :, ::-1])
            else:
                for i in range(down_t):
                    index = this_scene[i].split('/')[-1].split('.')[0]
                    cv2.imwrite(os.path.join(quan_p, f'{index}_quan.png'), LR_img[i][:, :, ::-1])
                    cv2.imwrite(os.path.join(latent_p, f'{index}_latent.png'), x_down[i][:, :, ::-1])
                    cv2.imwrite(os.path.join(rev_p, f'{index}_back.png'), rev_back[i][:, :, ::-1])

            # 计算指标并保存 SR 结果
            tmp_psnr_list = []
            tmp_ssim_list = []
            tmp_psnr_list_y = []
            tmp_ssim_list_y = []
            for i in range(7):
                base_indx = this_scene[i].split('/')[-1]
                sr_path = os.path.join(sr_p, base_indx)
                cv2.imwrite(sr_path, out[i][:, :, ::-1])

                gt_path = os.path.join(data_dir, this_scene[i])
                gt = cv2.imread(gt_path)
                sr_img = cv2.imread(sr_path)

                # 计算指标
                psnr_y = calculate_psnr(gt, sr_img, crop_border=0, test_y_channel=True)
                ssim_y = calculate_ssim(gt, sr_img, crop_border=0, test_y_channel=True)
                psnr = calculate_psnr(gt, sr_img, crop_border=0, test_y_channel=False)
                ssim = calculate_ssim(gt, sr_img, crop_border=0, test_y_channel=False)

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                tmp_psnr_list.append(psnr)
                tmp_ssim_list.append(ssim)

                psnr_list_y.append(psnr_y)
                ssim_list_y.append(ssim_y)
                tmp_psnr_list_y.append(psnr_y)
                tmp_ssim_list_y.append(ssim_y)

            # 打印当前序列的指标
            print(f'序列 psnr Y {sum(tmp_psnr_list_y)/len(tmp_psnr_list_y):.4f} '
                  f'序列 ssim Y {sum(tmp_ssim_list_y)/len(tmp_ssim_list_y):.4f}')
            print(f'序列 psnr {sum(tmp_psnr_list)/len(tmp_psnr_list):.4f} '
                  f'序列 ssim {sum(tmp_ssim_list)/len(tmp_ssim_list):.4f}')

    # 打印总体平均指标
    print(f'平均 psnr {sum(psnr_list)/len(psnr_list):.4f} '
          f'平均 ssim {sum(ssim_list)/len(ssim_list):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/path/to/your/dir/dataset/Vid4/GT_cp/',
                        help='Vid4 数据集目录路径')
    parser.add_argument('--base_out_p', type=str, 
                        default='/path/to/your/dir/code/ST_rescale_open_source/CSTVR/output',
                        help='结果保存的基础输出目录')
    parser.add_argument('--weight_base_p', type=str, 
                        default='/path/to/your/dir/code/ST_rescale/archived/',
                        help='模型权重的基础路径')
    parser.add_argument('--test_dataset_name', type=str, 
                        default='vid4',
                        help='测试数据集名称')
    
    args = parser.parse_args()
    test_vid4(args.data_dir, args.base_out_p, args.weight_base_p, args.test_dataset_name)

