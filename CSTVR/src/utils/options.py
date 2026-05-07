import argparse
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
import os
from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only

def yaml_load(f):
    """加载 yaml 文件或字符串。

    参数:
        f (str): 文件路径或 python 字符串。

    返回:
        dict: 加载后的字典。
    """
    
    with open(f, 'r') as f:
        return yaml.load(f, Loader=ordered_yaml()[0])
   
def ordered_yaml():
    """让 yaml 支持 OrderedDict。

    返回:
        yaml 加载器和转储器。
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """将字典转换为字符串，用于打印选项。

    参数:
        opt (dict): 选项字典。
        indent_level (int): 缩进级别。默认值: 1。

    返回:
        (str): 用于打印的选项字符串。
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    """后处理 yml 中的值，转换数据类型。"""
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    """解析命令行参数和配置文件。

    参数:
        root_path (str): 根路径。
        is_train (bool): 是否为训练模式。默认值: True。

    返回:
        opt (dict): 选项字典。
        args (argparse.Namespace): 命令行参数。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='../options/train/train_STSR_contin.yml', help='YAML 配置文件路径。')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='作业启动器')
    parser.add_argument('--auto_resume', action='store_true', help='自动恢复训练')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--local_rank', type=int, default=0) # 用于 pytorch 2.0
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='强制更新 yml 文件。示例: train:ema_decay=0.999')
    args = parser.parse_args()

    # 将 yml 解析为字典
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # 分布式设置
    if args.launcher == 'none':
        opt['dist'] = False
        print('禁用分布式训练。', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # 随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # 强制更新 yml 选项
    if args.force_yml is not None:
        for entry in args.force_yml:
            # 目前不支持创建新键
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # 使用 exec 函数
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # 调试模式设置
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # 数据集配置处理
    for phase, dataset in opt['datasets'].items():
        # 处理多个数据集，例如 val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']

    # 路径配置处理
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # 调试模式下修改某些选项
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # 测试模式
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    """复制选项文件到实验根目录。"""
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# 生成时间: {time.asctime()}\n# 命令:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
