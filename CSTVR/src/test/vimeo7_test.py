import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from os import path as osp
import numpy as np
from utils.model_utils import get_model_total_params
import data.core_bicubic as core_bicubic
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

def test_vimeo(data_dir, base_out_p, weight_base_p, test_dataset_name, time_factor, scale_factor):
    """ 在 Vimeo 数据集上测试 7 帧视频序列的重缩放。"""
    from data.vimeo_seq_dataset import Vimeo_SepTuplet
    from arch.IMSM import IND_inv3D
    from utils.options import yaml_load

    device = torch.device('cuda')
    train_dataset_name = 'vimeo'

    # 构建带参数的路径
    weight_p = os.path.join(weight_base_p, f'Tx{time_factor}_Sx{scale_factor}_{train_dataset_name}')
    out_p = os.path.join(base_out_p, f'Tx{time_factor}_Sx{scale_factor}', test_dataset_name)

    # 加载可逆神经网络模型 (Inverter)
    inv_root_p = os.path.join(weight_p, 'inverter/config.yml')
    inv_opt = yaml_load(inv_root_p)['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight_p = os.path.join(weight_p, 'inverter/model.pth')
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'], strict=True)

    # 加载重缩放模型 (Rescaler)
    model_root_path = os.path.join(weight_p, 'rescaler/config.yml')
    rescale_opt = yaml_load(model_root_path)
    if time_factor == 2 and scale_factor == 1:
        from arch.Mynet_arch import RescalerNet
    else:
        from arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to(device)
    rescale_weight_p = os.path.join(weight_p, 'rescaler/model.pth')
    weight = torch.load(rescale_weight_p)
    rescale_model.load_state_dict(weight['params'], strict=True)
    rescale_model.eval()
    param = get_model_total_params(rescale_model)
    print(f'参数量: {param}')

    # 数据集配置
    data_opt = rescale_opt['datasets']['val']
    dataset = Vimeo_SepTuplet(data_opt)
    avg_psnr = []
    avg_ssim = []
    avg_psnr_y = []
    avg_ssim_y = []

    avg_down_psnr_y = []
    avg_down_ssim_y = []
    avg_down_psnr = []
    avg_down_ssim = []
    
    # 确定下采样后的尺寸
    down_t = 7 if rescale_opt['down_type']['T'] == 1 else 4
    down_size = (down_t, 256 // rescale_opt['down_type']['S'], 448 // rescale_opt['down_type']['S'])

    for ix, data in enumerate(dataset):
        imgs = data['imgs'].unsqueeze(0).cuda()
        img_p = '/'.join(data['path'].split('/')[-2:])
        this_p = os.path.join(out_p, img_p)
        os.makedirs(this_p, exist_ok=True)
        print(img_p)
        with torch.no_grad():
            # 下采样过程
            x_down = rescale_model.inference_down(imgs, down_size)
            # 潜在特征转 RGB (隐写)
            LR_img = model.inference_latent2RGB(x_down)
            LR_img = LR_img.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            LR_img = LR_img.astype(np.uint8)
            LR_img_ten = torch.from_numpy(LR_img).unsqueeze(0).permute(0, 4, 1, 2, 3).cuda() / 255.0

            # 逆向重建过程
            rev_back = model.inference_RGB2latent(LR_img_ten)
            out = rescale_model.inference_up(rev_back, (imgs.shape[2], imgs.shape[3], imgs.shape[4]))

            # 转换为 numpy 以便保存
            x_down = x_down.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            out = out.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            rev_back = rev_back.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0

            # 创建子目录
            quan_p = os.path.join(this_p, 'quant')
            latent_p = os.path.join(this_p, 'latent')
            rev_p = os.path.join(this_p, 'rev')
            sr_p = os.path.join(this_p, 'sr')
            os.makedirs(quan_p, exist_ok=True)
            os.makedirs(latent_p, exist_ok=True)
            os.makedirs(rev_p, exist_ok=True)
            os.makedirs(sr_p, exist_ok=True)
            
            seq_down_psnr = []
            seq_down_ssim = []
            seq_down_psnr_y = []
            seq_down_ssim_y = []
            
            # 处理下采样结果保存与指标计算
            if down_t == 7:
                for i in range(down_t):
                    cv2.imwrite(quan_p+'/im'+str(i+1)+'_quan.png', LR_img[i][:, :, ::-1])
                    cv2.imwrite(latent_p+'/im'+str(i+1)+'_latent.png', x_down[i][:, :, ::-1])
                    cv2.imwrite(rev_p+'/im'+str(i+1)+'_back.png', rev_back[i][:, :, ::-1])

                    gt_p = data['path'] + '/' + 'im' + str(i + 1) + '.png'
                    gt_img = cv2.imread(gt_p)
                    
                    down_img = cv2.imread(quan_p+'/im'+str(i+1)+'_quan.png')
                    if args.scale_factor > 1:
                        gt_img = gt_img.astype('float32')
                        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)
                        h, w = 256, 448
                        gt_img = core_bicubic.imresize(gt_img.contiguous(), sizes=(h // args.scale_factor, w // args.scale_factor))
                        gt_img = gt_img.squeeze(0).permute(1, 2, 0).numpy()

                    psnr_y = calculate_psnr(gt_img, down_img, crop_border=0, test_y_channel=True)
                    ssim_y = calculate_ssim(gt_img, down_img, crop_border=0, test_y_channel=True)
                    psnr = calculate_psnr(gt_img, down_img, crop_border=0, test_y_channel=False)
                    ssim = calculate_ssim(gt_img, down_img, crop_border=0, test_y_channel=False)
                    
                    seq_down_psnr.append(psnr)
                    seq_down_ssim.append(ssim)
                    seq_down_psnr_y.append(psnr_y)
                    seq_down_ssim_y.append(ssim_y)

                    avg_down_psnr.append(psnr)
                    avg_down_ssim.append(ssim)
                    avg_down_psnr_y.append(psnr_y)
                    avg_down_ssim_y.append(ssim_y)
                    print(f'下采样 {img_p}/im{str(i+1)}.png psnr {psnr} ssim {ssim} psnr Y {psnr_y} ssim Y {ssim_y}')
            else:
                for i in range(down_t):
                    cv2.imwrite(quan_p+'/im'+str(2*i+1)+'_quan.png', LR_img[i][:, :, ::-1])
                    cv2.imwrite(latent_p+'/im'+str(2*i+1)+'_latent.png', x_down[i][:, :, ::-1])
                    cv2.imwrite(rev_p+'/im'+str(2*i+1)+'_back.png', rev_back[i][:, :, ::-1])

                    gt_p = data['path'] + '/' + 'im' + str(2 * i + 1) + '.png'
                    gt_img = cv2.imread(gt_p)
                    
                    down_img = cv2.imread(quan_p+'/im'+str(2*i+1)+'_quan.png')
                    if args.scale_factor > 1:
                        gt_img = gt_img.astype('float32')
                        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)
                        h, w = 256, 448
                        gt_img = core_bicubic.imresize(gt_img.contiguous(), sizes=(h // args.scale_factor, w // args.scale_factor))
                        gt_img = gt_img.squeeze(0).permute(1, 2, 0).numpy()
                    
                    psnr_y = calculate_psnr(gt_img, down_img, crop_border=0, test_y_channel=True)
                    ssim_y = calculate_ssim(gt_img, down_img, crop_border=0, test_y_channel=True)
                    psnr = calculate_psnr(gt_img, down_img, crop_border=0, test_y_channel=False)
                    ssim = calculate_ssim(gt_img, down_img, crop_border=0, test_y_channel=False)
                    
                    if psnr_y < 90:
                        seq_down_psnr.append(psnr)
                        seq_down_ssim.append(ssim)
                        seq_down_psnr_y.append(psnr_y)
                        seq_down_ssim_y.append(ssim_y)

                        avg_down_psnr.append(psnr)
                        avg_down_ssim.append(ssim)
                        avg_down_psnr_y.append(psnr_y)
                        avg_down_ssim_y.append(ssim_y)
                    else:
                        continue
                   
                    print(f'下采样 {img_p}/im{str(2*i+1)}.png psnr {psnr} ssim {ssim} psnr Y {psnr_y} ssim Y {ssim_y}')
                if len(seq_down_psnr) > 0:
                    print(f'平均下采样 {img_p} psnr {sum(seq_down_psnr)/len(seq_down_psnr)} ssim {sum(seq_down_ssim)/len(seq_down_ssim)}')
                    print(f'平均下采样 {img_p} psnr Y {sum(seq_down_psnr_y)/len(seq_down_psnr_y)} ssim Y {sum(seq_down_ssim_y)/len(seq_down_ssim_y)}')
            
            # 处理最终 SR 结果保存与指标计算
            seq_psnr = []
            seq_ssim = []
            seq_psnr_y = []
            seq_ssim_y = []
     
            for i in range(7):
                cv2.imwrite(sr_p+'/im'+str(i+1)+'_out.png', out[i][:, :, ::-1])
                gt_p = data['path'] + '/' + 'im' + str(i + 1) + '.png'
                gt = cv2.imread(gt_p)
                this_img = cv2.imread(sr_p+'/im'+str(i+1)+'_out.png')
           
                psnr = calculate_psnr(gt, this_img, crop_border=0, test_y_channel=False)
                ssim = calculate_ssim(gt, this_img, crop_border=0, test_y_channel=False)
                psnr_y = calculate_psnr(gt, this_img, crop_border=0, test_y_channel=True)
                ssim_y = calculate_ssim(gt, this_img, crop_border=0, test_y_channel=True)
                
                if psnr < 90:
                    avg_psnr.append(psnr)
                    seq_psnr.append(psnr)
                    avg_psnr_y.append(psnr_y)
                    seq_psnr_y.append(psnr_y)
                    
                    avg_ssim.append(ssim)
                    seq_ssim.append(ssim)
                    avg_ssim_y.append(ssim_y)
                    seq_ssim_y.append(ssim_y)
                    print(f'ix {i+1} psnr {psnr} ssim {ssim} psnr_Y {psnr_y} ssim_Y {ssim_y}')
            
            if len(seq_psnr) > 0:
                print(f'序列 psnr {sum(seq_psnr)/len(seq_psnr)} ssim {sum(seq_ssim)/len(seq_ssim)} psnr_y {sum(seq_psnr_y)/len(seq_psnr_y)} ssim_y {sum(seq_ssim_y)/len(seq_ssim_y)}')
            print(f'目前平均 psnr {sum(avg_psnr)/len(avg_psnr)} ssim {sum(avg_ssim)/len(avg_ssim)} 平均 psnr_y {sum(avg_psnr_y)/len(avg_psnr_y)} ssim_y {sum(avg_ssim_y)/len(avg_ssim_y)} ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='path/to/your/dir/dataset/Vid4/GT_cp/',
                        help='Vimeo 数据集目录路径')
    parser.add_argument('--base_out_p', type=str, 
                        default='path/to/your/dir/code/ST_rescale_open_source/CSTVR/output',
                        help='结果保存的基础输出目录')
    parser.add_argument('--weight_base_p', type=str, 
                        default='path/to/your/dir/code/ST_rescale_open_source/CSTVR/archived/',
                        help='模型权重的基础路径')
    parser.add_argument('--test_dataset_name', type=str, 
                        default='vimeo',
                        help='测试数据集名称')
    parser.add_argument('--time_factor', type=int, 
                        default=2,
                        help='时间缩放因子')
    parser.add_argument('--scale_factor', type=int, 
                        default=1,
                        help='空间缩放因子')
    
    args = parser.parse_args()
    test_vimeo(args.data_dir, args.base_out_p, args.weight_base_p, 
               args.test_dataset_name, args.time_factor, args.scale_factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='path/to/your/dir/dataset/Vid4/GT_cp/',
                        help='Path to Vimeo dataset directory')
    parser.add_argument('--base_out_p', type=str, 
                        default='path/to/your/dir/code/ST_rescale_open_source/CSTVR/output',
                        help='Base output directory for results')
    parser.add_argument('--weight_base_p', type=str, 
                        default='path/to/your/dir/code/ST_rescale_open_source/CSTVR/archived/',
                        help='Base path for model weights')
    parser.add_argument('--test_dataset_name', type=str, 
                        default='vimeo',
                        help='Name of test dataset')
    parser.add_argument('--time_factor', type=int, 
                        default=2,
                        help='Temporal scaling factor')
    parser.add_argument('--scale_factor', type=int, 
                        default=1,
                        help='Spatial scaling factor')
    
    args = parser.parse_args()
    test_vimeo(args.data_dir, args.base_out_p, args.weight_base_p, 
               args.test_dataset_name, args.time_factor, args.scale_factor)