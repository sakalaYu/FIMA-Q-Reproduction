import os
import sys
import torch
from torch import nn
import numpy as np
from functools import partial
import argparse
import importlib
import timm
import copy
import time
import re
import shutil
import random
import json
import math
import hashlib

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import *
from utils.orthogonal_vit import apply_headwise_hadamard
from datetime import datetime
import logging

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        run_name = os.environ.get('FIMAQ_RUN_NAME', '').strip()
        run_name = re.sub(r'[^A-Za-z0-9._-]+', '_', run_name).strip('._-')
        folder_name = formatted_timestamp if not run_name else '{}_{}'.format(formatted_timestamp, run_name)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(project_dir, 'checkpoints', 'quant_result', folder_name)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(1)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])


import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/vit_config.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="/dataset/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--optim-size", default=1024,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--optim-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")

    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true', help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None, help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument("--calib-metric", type=str, default=argparse.SUPPRESS, choices=['mse', 'mae'], 
                        help='calibration metric')
    parser.add_argument("--optim-metric", type=str, default=argparse.SUPPRESS, choices=['fisher_brecq', 'fisher_lr', 'fisher_diag', 'fisher_dplr', 'forward_secant', 'robust_fisher', 'mse', 'mae'],
                        help='optimization metric')
    parser.add_argument('--optim-mode', type=str, default=argparse.SUPPRESS, choices=['qinp', 'rinp', 'qdrop'], 
                        help='`qinp`:use quanted input; `rinp`: use raw input; `qdrop` use qdrop input;')
    parser.add_argument('--drop-prob', type=float, default=argparse.SUPPRESS, 
                        help='dropping rate in qdrop. set `drop-prob = 1.0` if do not use qdrop.')
    parser.add_argument('--k', type=int, default=argparse.SUPPRESS, help='The rank of Fisher')
    parser.add_argument('--p1', type=float, default=argparse.SUPPRESS, help='The proportion of low rank')
    parser.add_argument('--p2', type=float, default=argparse.SUPPRESS, help='The proportion of diag')
    parser.add_argument('--robust-beta', type=float, default=argparse.SUPPRESS,
                        help='weight of smooth worst-group loss (0 = balanced ERM)')
    parser.add_argument('--robust-temperature', type=float, default=argparse.SUPPRESS,
                        help='temperature of smooth worst-group loss')
    parser.add_argument('--dis-mode', type=str, default=argparse.SUPPRESS, choices=['q','qf'],
                        help='the mode of getting gradient. `q`: use quantization; `qf` Take the first k times (default:Uniformly obtain k times);')
    parser.add_argument('--recon-iters', type=int, default=argparse.SUPPRESS,
                        help='fixed reconstruction iterations per block')
    parser.add_argument('--skip-final-validation', action='store_true',
                        help='skip calibration/test validation for a short integration smoke run')
    parser.add_argument('--headwise-hadamard', action='store_true',
                        help='apply fixed, function-preserving Hadamard rotations before quantization')
    parser.add_argument('--headwise-mode', choices=('all', 'qk', 'vo'), default='all',
                        help='rotation location; requires --headwise-hadamard for qk/vo')
    parser.add_argument('--qk-joint-variant', choices=('none', 'A', 'B', 'C'), default='none')
    parser.add_argument('--qk-joint-protocol', choices=('v2_integer_zp', 'v3_lowrank_gate'), default='v2_integer_zp')
    parser.add_argument('--qk-joint-steps', type=int, default=300)
    parser.add_argument('--qk-joint-batch-size', type=int, default=8)
    parser.add_argument('--qk-joint-holdout', type=int, default=128)
    parser.add_argument('--qk-joint-lr', type=float, default=0.003)
    parser.add_argument('--qk-joint-eval-every', type=int, default=50)
    parser.add_argument('--qk-joint-rotation-rank', type=int, default=0)
    parser.add_argument('--qk-joint-min-relative-kl', type=float, default=0.0)
    parser.add_argument('--qk-joint-reconstruction-loss', choices=('auto', 'mse', 'dplr'), default='auto',
                        help='pre-calibration block loss; original full reconstruction remains DPLR')
    parser.add_argument('--attention-relation', choices=('none', 'uniform', 'fisher', 'aligned'), default='none')
    parser.add_argument('--relation-protocol', choices=('relation_v1', 'relation_v2_alignment'), default='relation_v1')
    parser.add_argument('--relation-weight', type=float, default=0.1)
    parser.add_argument('--relation-queries', type=int, default=16)
    parser.add_argument('--relation-start', type=float, default=0.2)
    parser.add_argument('--relation-full', type=float, default=0.6)
    parser.add_argument('--probe-kd', choices=('none', 'uniform', 'reliable'), default='none')
    parser.add_argument('--probe-kd-protocol', choices=('semantic_probe_v1',), default='semantic_probe_v1')
    parser.add_argument('--probe-kd-weight', type=float, default=0.05)
    parser.add_argument('--probe-kd-temperature', type=float, default=2.0)
    parser.add_argument('--probe-kd-ridge', type=float, default=0.1)
    parser.add_argument('--probe-kd-holdout', type=int, default=128)
    parser.add_argument('--probe-kd-code-sha', default='unrecorded')
    parser.add_argument('--endpoint-task', choices=('none', 'rank10', 'hard-repair', 'qwt', 'soft-control', 'diagnose'), default='none')
    parser.add_argument('--endpoint-code-sha', default='unrecorded')
    parser.add_argument('--endpoint-mode', choices=('none', 'hard_scale', 'soft_extra', 'diagnose'), default='none')
    parser.add_argument('--endpoint-steps', type=int, default=2000)
    parser.add_argument('--endpoint-lr', type=float, default=0.001,
                        help='Adam LR for log activation-scale multipliers (hard repair only)')
    parser.add_argument('--endpoint-bound', type=float, default=1.2)
    parser.add_argument('--endpoint-holdout', type=int, default=128)
    parser.add_argument('--endpoint-batch-size', type=int, default=8)
    parser.add_argument('--endpoint-eval-every', type=int, default=200)
    parser.add_argument('--qwt', action='store_true')
    parser.add_argument('--qwt-samples', type=int, default=512)
    parser.add_argument('--qwt-batch-size', type=int, default=8)
    parser.add_argument('--qwt-ridge', type=float, default=0.0)
    parser.add_argument('--qwt-branch-dtype', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--qwt-solver-threads', type=int, default=4)
    parser.add_argument('--activation-codebook', choices=('none', 'softmax', 'gelu'), default='none')
    parser.add_argument('--method-adaptation', choices=('none', 'kd', 'adalog', 'erq', 'lsvit', 'qkcode', 'qkcode_kd'), default='none')
    parser.add_argument('--lsvit-control', choices=('c', 'plain', 'fixed', 'b'), default='c',
                        help='LS recipe: legacy C-MSE, no rotation, fixed rotation, or B-MSE')
    parser.add_argument('--adaptation-steps', type=int, default=500)
    return parser


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate', tag=''):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_calibsize_{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size, cfg.calib_metric)
    else:
        auto_name = '{}_w{}_a{}_optimsize_{}_{}{}{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, cfg.optim_metric, 
            '' if cfg.optim_metric in ['mse', 'mae'] else '_dis_mode_' + cfg.dis_mode, 
            '' if cfg.optim_metric not in['fisher_lr', 'fisher_dplr', 'forward_secant', 'robust_fisher'] else '_rank_' + str(cfg.k),
            cfg.optim_mode)
    if tag:
        auto_name = auto_name[:-4] + '_' + tag + '.pth'
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)
    return save_path


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if any(key.endswith('.adalog_q') for key in ckpt):
        from utils.method_adaptations import attach_method_quantizers
        attach_method_quantizers(model, ckpt)
    has_qwt = any('.qwt_weight' in key for key in ckpt)
    has_codebooks = any(key.endswith('.codebook') for key in ckpt)
    if getattr(args, 'activation_codebook', 'none') != 'none' and (has_qwt or has_codebooks):
        raise ValueError('Codebook fitting requires the original C-MSE checkpoint before compensation')
    if has_qwt:
        from utils.qwt_compensation import attach_saved_qwt
        attach_saved_qwt(model, ckpt)
    if has_codebooks:
        from utils.activation_codebook import attach_saved_codebooks
        attach_saved_codebooks(model, ckpt)
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, 'scale'):
                quantizer.inited = True
                ckpt_name = name + '.' + attr + '.scale'
                quantizer.scale.data = ckpt[ckpt_name].clone()
 
    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model

def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info("Experiment directory: {}".format(root_path))
    logging.info(str(args))

    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")
        
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.optim_size = args.optim_size if hasattr(args, 'optim_size') else cfg.optim_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.optim_batch_size = args.optim_batch_size if hasattr(args, 'optim_batch_size') else cfg.optim_batch_size
    cfg.calib_metric = args.calib_metric if hasattr(args, 'calib_metric') else cfg.calib_metric
    cfg.optim_metric = args.optim_metric if hasattr(args, 'optim_metric') else cfg.optim_metric
    cfg.optim_mode = args.optim_mode if hasattr(args, 'optim_mode') else cfg.optim_mode
    cfg.drop_prob = args.drop_prob if hasattr(args, 'drop_prob') else cfg.drop_prob
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    cfg.k = args.k if hasattr(args, 'k') else cfg.k
    cfg.p1 = args.p1 if hasattr(args, 'p1') else cfg.p1
    cfg.p2 = args.p2 if hasattr(args, 'p2') else cfg.p2
    cfg.robust_beta = args.robust_beta if hasattr(args, 'robust_beta') else getattr(cfg, 'robust_beta', 0.5)
    cfg.robust_temperature = (args.robust_temperature if hasattr(args, 'robust_temperature')
                              else getattr(cfg, 'robust_temperature', 0.5))
    cfg.dis_mode = args.dis_mode if hasattr(args, 'dis_mode') else cfg.dis_mode
    cfg.recon_iters = args.recon_iters if hasattr(args, 'recon_iters') else getattr(cfg, 'recon_iters', 20000)
    if args.method_adaptation == 'lsvit':
        cfg.optim_metric = 'lsvit'
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")
        
    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        args.device = 'cuda:0'
    device = torch.device(args.device)
    
    model_zoo = {
        'vit_tiny'  : 'vit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',
    }

    seed_all(args.seed)
    logging.info('Python random seeded: %d', args.seed)
    
    logging.info('Building model ...')
    try:
        model = timm.create_model(model_zoo[args.model], checkpoint_path='./checkpoints/vit_raw/{}.bin'.format(model_zoo[args.model]))
    except:
        model = timm.create_model(model_zoo[args.model], pretrained=True)

    if args.endpoint_task != 'none' or args.method_adaptation != 'none':
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode('utf-8'))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        model_sha = digest.hexdigest()
        logging.info('Endpoint FP model SHA256: %s', model_sha)
    else:
        model_sha = None
    model.to(device)
    model.eval()
    adaptation_teacher = None
    if args.method_adaptation in ('kd', 'adalog', 'erq', 'qkcode', 'qkcode_kd'):
        adaptation_teacher = copy.deepcopy(model).cpu().eval()
        adaptation_teacher.requires_grad_(False)
        if args.method_adaptation == 'erq':
            apply_headwise_hadamard(adaptation_teacher, verify=True, mode='all')
    codebook_teacher = None
    if args.activation_codebook != 'none':
        if cfg.w_bit != 3 or cfg.a_bit != 3:
            raise ValueError('Activation codebook experiments require W3A3')
        codebook_teacher = copy.deepcopy(model).cpu().eval()
    if args.headwise_hadamard:
        rotation_info = apply_headwise_hadamard(model, verify=True, mode=args.headwise_mode)
        logging.info("Applied head-wise Hadamard rotation: {}".format(rotation_info))
    qwt_teacher = copy.deepcopy(model).cpu().eval() if args.qwt and not args.load_optimize_checkpoint else None
    if qwt_teacher is not None:
        qwt_teacher.requires_grad_(False)
    endpoint_result = dict(task=args.endpoint_task, code_sha256=args.endpoint_code_sha,
                           fp_model_sha256=model_sha, args=vars(args), cfg=vars(cfg))
    if args.endpoint_task != 'none':
        endpoint_result['launch_identity'] = json.loads(os.environ.get('FIMAQ_ENDPOINT_IDENTITY', 'null'))
    data_path = args.dataset
    g = mydatasets.ViTImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model":model})
    
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)
   
    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wraping quantiztion modules (reparam: {}) ...'.format(reparam))
    
    model = wrap_modules_in_net(model, cfg, reparam=reparam)
    model.to(device)
    model.eval()

    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode='calibrate')
            if args.test_calibrate_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        else:
            logging.info("{} - start {} guided calibration".format(get_cur_time(), cfg.calib_metric))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - {} guided calibration finished.".format(get_cur_time(), cfg.calib_metric))
            save_model(model, args, cfg, mode='calibrate')
            if not args.skip_final_validation:
                logging.info('Validating after calibration ...')
                val_loss, val_prec1, val_prec5 = validate(
                    val_loader, model, criterion, print_freq=args.print_freq, device=device
                )

    if args.optimize:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logging.info("{} - start {} guided block reconstruction".format(get_cur_time(), cfg.optim_metric))
        qk_options = None
        if args.qk_joint_variant != 'none':
            if cfg.optim_metric not in ('fisher_dplr', 'lsvit'):
                raise ValueError('QK joint calibration requires fisher_dplr reconstruction')
            if not 0 < args.qk_joint_holdout < cfg.optim_size:
                raise ValueError('QK joint holdout must be between 1 and optim_size - 1')
            qk_options = dict(variant=args.qk_joint_variant, protocol=args.qk_joint_protocol,
                              reconstruction_loss=args.qk_joint_reconstruction_loss,
                              rotation_rank=args.qk_joint_rotation_rank,
                              min_relative_kl=args.qk_joint_min_relative_kl,
                              steps=args.qk_joint_steps,
                              batch_size=args.qk_joint_batch_size, holdout=args.qk_joint_holdout,
                              lr=args.qk_joint_lr, eval_every=args.qk_joint_eval_every, seed=args.seed)
        relation_options = None
        if args.attention_relation != 'none':
            if cfg.optim_metric != 'fisher_dplr':
                raise ValueError('Attention relations require fisher_dplr reconstruction')
            relation_options = dict(mode=args.attention_relation, weight=args.relation_weight,
                                    rows=args.relation_queries, start=args.relation_start,
                                    full=args.relation_full, protocol=args.relation_protocol)
        probe_options = None
        if args.probe_kd != 'none':
            probe_options = dict(mode=args.probe_kd, weight=args.probe_kd_weight,
                                 temperature=args.probe_kd_temperature, ridge=args.probe_kd_ridge,
                                 holdout=args.probe_kd_holdout, protocol=args.probe_kd_protocol)
        endpoint_options = None
        if args.endpoint_mode != 'none':
            endpoint_options = dict(mode=args.endpoint_mode, steps=args.endpoint_steps,
                lr=args.endpoint_lr, bound=args.endpoint_bound, holdout=args.endpoint_holdout,
                batch_size=args.endpoint_batch_size, eval_every=args.endpoint_eval_every, seed=args.seed)
        block_reconstructor = BlockReconstructor(model, cfg.optim_batch_size, calib_loader,
            metric=cfg.optim_metric, temp=cfg.temp, k=cfg.k, dis_mode=cfg.dis_mode,
            p1=cfg.p1, p2=cfg.p2, iters=cfg.recon_iters,
            robust_beta=cfg.robust_beta, robust_temperature=cfg.robust_temperature,
            qk_joint_options=qk_options, attention_relation_options=relation_options,
            probe_kd_options=probe_options, endpoint_options=endpoint_options)
        reconstruction_started = time.perf_counter()
        block_reconstructor.reconstruct_model(quant_act=True, mode=cfg.optim_mode, drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        logging.info('Block reconstruction elapsed seconds: {:.3f}'.format(
            time.perf_counter() - reconstruction_started))
        logging.info("{} - {} guided block reconstruction finished.".format(get_cur_time(), cfg.optim_metric))
        if qk_options:
            endpoint_result['qk_blocks'] = block_reconstructor.qk_joint_results
            with open(os.path.join(root_path, 'qk_joint.json'), 'w', encoding='utf-8') as handle:
                json.dump(dict(options=qk_options, blocks=block_reconstructor.qk_joint_results), handle, indent=2)
        if relation_options:
            with open(os.path.join(root_path, 'attention_relation.json'), 'w', encoding='utf-8') as handle:
                json.dump(dict(options=relation_options, blocks=block_reconstructor.attention_relation_results), handle, indent=2)
        if probe_options:
            with open(os.path.join(root_path, 'probe_kd.json'), 'w', encoding='utf-8') as handle:
                json.dump(dict(options=probe_options, blocks=block_reconstructor.probe_kd_results), handle, indent=2)
        if endpoint_options:
            endpoint_result['endpoint'] = dict(options=endpoint_options, blocks=block_reconstructor.endpoint_results)
            with open(os.path.join(root_path, 'endpoint_blocks.json'), 'w', encoding='utf-8') as handle:
                json.dump(endpoint_result['endpoint'], handle, indent=2)
        endpoint_result['baseline_checkpoint'] = save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode='optimize')
    if (args.optimize or args.test_optimize_checkpoint) and not args.skip_final_validation:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(calib_loader, model, criterion, print_freq=args.print_freq, device=device)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        endpoint_result['final'] = dict(top1=float(val_prec1), top5=float(val_prec5), loss=float(val_loss))
    if args.method_adaptation in ('kd', 'adalog', 'erq', 'qkcode', 'qkcode_kd'):
        from utils.method_adaptations import adapt_model
        before = dict(endpoint_result['final'])
        report = adapt_model(model, adaptation_teacher, calib_loader, args.method_adaptation, args.adaptation_steps)
        del adaptation_teacher
        report['before'] = before
        report['checkpoint_before'] = args.load_optimize_checkpoint
        logging.info('Validating after method adaptation ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        report['after'] = dict(top1=float(val_prec1), top5=float(val_prec5), loss=float(val_loss))
        report['top1_delta_pp'] = report['after']['top1'] - before['top1']
        report['checkpoint_after'] = save_model(model, args, cfg, mode='optimize', tag=args.method_adaptation)
        report['launch'] = json.loads(os.environ.get('FIMAQ_METHOD_LAUNCH', '{}'))
        logging.info('Method adaptation result: %s', json.dumps(report))
    elif args.method_adaptation == 'lsvit':
        report = dict(method=('lsvit_C_mse_streaming_damped' if args.lsvit_control == 'c'
                              else 'lsvit_' + args.lsvit_control + '_streaming_damped'),
                      control=args.lsvit_control, args=vars(args), cfg=vars(cfg),
                      checkpoint_after=endpoint_result['baseline_checkpoint'],
                      after=endpoint_result['final'],
                      qk_blocks=block_reconstructor.qk_joint_results,
                      launch=json.loads(os.environ.get('FIMAQ_METHOD_LAUNCH', '{}')))
    if args.method_adaptation != 'none':
        report['fp_model_sha256'] = model_sha
        report['calibration_hashes'] = re.findall(
            r'Calibration cache: count=(\d+) sample_sha256=([a-f0-9]+) tensor_sha256=([a-f0-9]+)',
            open(os.path.join(root_path, 'output.log'), encoding='utf-8').read())
        report['status'] = 'complete'
        with open(os.path.join(root_path, 'method_result.json.tmp'), 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        os.replace(os.path.join(root_path, 'method_result.json.tmp'), os.path.join(root_path, 'method_result.json'))
    if codebook_teacher is not None:
        from utils.activation_codebook import fit_activation_codebooks
        report = fit_activation_codebooks(model, codebook_teacher, calib_loader, args.activation_codebook)
        del codebook_teacher
        report['checkpoint_before'] = args.load_optimize_checkpoint
        report['before'] = endpoint_result['final']
        logging.info('Validating on test set after activation codebook fitting ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        report['after'] = dict(top1=float(val_prec1), top5=float(val_prec5), loss=float(val_loss))
        report['top1_delta_pp'] = report['after']['top1'] - report['before']['top1']
        report['checkpoint_after'] = save_model(model, args, cfg, mode='optimize', tag=args.activation_codebook + '_codebook')
        path = os.path.join(root_path, 'activation_codebook_result.json')
        with open(path + '.tmp', 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        os.replace(path + '.tmp', path)
        logging.info('Activation codebook paired Top-1 delta: %+.3f pp', report['top1_delta_pp'])
    if args.qwt and not args.load_optimize_checkpoint:
        from utils.qwt_compensation import fit_qwt
        endpoint_result['before_qwt'] = endpoint_result.get('final')
        logging.info('Starting QwT compensation on the frozen reconstructed model ...')
        qwt_report = fit_qwt(model, qwt_teacher, calib_loader, samples=args.qwt_samples,
            batch_size=args.qwt_batch_size, ridge=args.qwt_ridge,
            branch_dtype=args.qwt_branch_dtype, solver_threads=args.qwt_solver_threads)
        endpoint_result['qwt'] = qwt_report
        with open(os.path.join(root_path, 'qwt.json'), 'w', encoding='utf-8') as handle:
            json.dump(qwt_report, handle, indent=2)
        del qwt_teacher
        endpoint_result['qwt_checkpoint'] = save_model(model, args, cfg, mode='optimize', tag='qwt')
        if not args.skip_final_validation:
            logging.info('Validating on test set after QwT compensation ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
            endpoint_result['final'] = dict(top1=float(val_prec1), top5=float(val_prec5), loss=float(val_loss))
            logging.info('QwT paired Top-1 delta: %+.3f pp',
                         endpoint_result['final']['top1'] - endpoint_result['before_qwt']['top1'])
    logging.info("{} - finished the process.".format(get_cur_time()))
    for handler in logging.getLogger().handlers:
        handler.flush()
    logs_dir = os.path.join(project_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(root_path, 'output.log'),
        os.path.join(logs_dir, os.path.basename(root_path) + '.log')
    )
    if args.endpoint_task != 'none':
        endpoint_result['status'] = 'smoke_complete' if args.skip_final_validation else 'complete'
        endpoint_result['run_id'] = os.path.basename(root_path)
        endpoint_result['calibration_hashes'] = re.findall(
            r'Calibration cache: count=(\d+) sample_sha256=([a-f0-9]+) tensor_sha256=([a-f0-9]+)',
            open(os.path.join(root_path, 'output.log'), encoding='utf-8').read())
        target = os.path.join(root_path, 'endpoint_result.json')
        with open(target + '.tmp', 'w', encoding='utf-8') as handle:
            json.dump(endpoint_result, handle, indent=2)
        os.replace(target + '.tmp', target)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    from utils.lsvit_protocol import validate_lsvit_control
    try:
        validate_lsvit_control(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.method_adaptation != 'none':
        if args.model != 'vit_small' or args.skip_final_validation or args.activation_codebook != 'none':
            parser.error('Method adaptations require ViT-S and complete validation')
        if args.adaptation_steps < 1 or args.optim_size != 1024:
            parser.error('Method adaptations require positive steps and 1024 calibration images')
        if args.method_adaptation != 'lsvit' and (
                not args.load_optimize_checkpoint or not args.test_optimize_checkpoint):
            parser.error('KD/AdaLog/ERQ require an optimized C-MSE checkpoint and paired validation')
    if args.activation_codebook != 'none' and (
            args.model != 'vit_small' or not args.load_optimize_checkpoint or not args.test_optimize_checkpoint
            or args.skip_final_validation or args.optim_size != 128 or args.endpoint_task != 'none'):
        parser.error('Activation codebook fitting requires ViT-S, an optimized checkpoint, full validation, and --optim-size 128')
    if args.endpoint_mode != 'none' or args.qwt or args.endpoint_task != 'none':
        if args.endpoint_task == 'none' or args.endpoint_code_sha == 'unrecorded':
            parser.error('Endpoint experiments require a task and source fingerprint; use run_endpoint_experiments.py')
        if not args.optimize or args.load_calibrate_checkpoint or args.load_optimize_checkpoint:
            parser.error('Endpoint experiments currently start from fresh C-MSE reconstruction')
        if args.qk_joint_variant != 'C' or args.qk_joint_reconstruction_loss != 'mse':
            parser.error('Endpoint experiments require C-MSE pre-calibration')
        if args.probe_kd != 'none' or args.attention_relation != 'none':
            parser.error('Endpoint experiments exclude KD and attention auxiliary losses')
        if min(args.endpoint_steps, args.endpoint_holdout, args.endpoint_batch_size,
               args.endpoint_eval_every, args.qwt_samples, args.qwt_batch_size, args.qwt_solver_threads) < 1:
            parser.error('Endpoint sizes and steps must be positive')
        if not 0 < args.endpoint_holdout < args.optim_size or (args.qwt and args.qwt_samples > args.optim_size):
            parser.error('Endpoint holdout/QwT samples exceed calibration size')
        if not math.isfinite(args.endpoint_lr) or args.endpoint_lr <= 0 or not math.isfinite(args.endpoint_bound) or args.endpoint_bound <= 1:
            parser.error('Repair LR must be positive and scale bound > 1')
        if not math.isfinite(args.qwt_ridge) or args.qwt_ridge < 0:
            parser.error('QwT ridge must be finite and nonnegative')
    if args.probe_kd != 'none':
        if args.qk_joint_variant not in ('B', 'C') or args.attention_relation != 'none':
            parser.error('Probe KD requires the B/C QK reconstruction path without attention relations')
        if args.model not in ('vit_small', 'vit_base', 'deit_small', 'deit_base', 'deit_tiny', 'vit_tiny'):
            parser.error('Probe KD requires ViT/DeiT CLS-token models')
        if not all(math.isfinite(x) and x > 0 for x in
                   (args.probe_kd_weight, args.probe_kd_temperature, args.probe_kd_ridge)):
            parser.error('Probe KD weight, temperature and ridge must be finite and positive')
        if not 0 < args.probe_kd_holdout < args.optim_size:
            parser.error('Probe KD holdout must be smaller than optim-size')
    if args.headwise_mode != 'all' and not args.headwise_hadamard:
        parser.error('--headwise-mode qk/vo requires --headwise-hadamard')
    if args.qk_joint_variant != 'none':
        if not args.headwise_hadamard or args.headwise_mode != 'all' or not args.optimize:
            parser.error('QK joint A/B/C requires --headwise-hadamard --headwise-mode all --optimize')
        if args.load_calibrate_checkpoint or args.load_optimize_checkpoint:
            parser.error('QK joint A/B/C starts from a fresh floating-point model')
        if args.model not in ('vit_small', 'vit_base', 'deit_small', 'deit_base', 'deit_tiny', 'vit_tiny'):
            parser.error('QK joint A/B/C currently supports timm ViT/DeiT models')
        if min(args.qk_joint_steps, args.qk_joint_batch_size, args.qk_joint_holdout,
               args.qk_joint_eval_every) < 1 or args.qk_joint_lr <= 0:
            parser.error('QK joint sizes, steps, interval and learning rate must be positive')
    if args.attention_relation != 'none':
        expected_protocol = ('relation_v2_alignment' if args.attention_relation == 'aligned'
                             else 'relation_v1')
        if args.relation_protocol != expected_protocol:
            parser.error('Attention relation mode and --relation-protocol do not match')
        if args.qk_joint_variant != 'none':
            parser.error('Run attention relations separately from QK pre-calibration')
        if not args.optimize or not args.headwise_hadamard or args.headwise_mode != 'all':
            parser.error('Attention relations require --optimize --headwise-hadamard --headwise-mode all')
        if args.load_calibrate_checkpoint or args.load_optimize_checkpoint:
            parser.error('This attention-relation protocol starts from fresh FP calibration')
        if args.model not in ('vit_small', 'vit_base', 'deit_small', 'deit_base', 'deit_tiny', 'vit_tiny'):
            parser.error('Attention relations currently support timm ViT/DeiT models')
        if args.relation_queries < 1 or not math.isfinite(args.relation_weight) or args.relation_weight <= 0:
            parser.error('Relation query count and finite weight must be positive')
        if not 0 <= args.relation_start < args.relation_full <= 1:
            parser.error('Require 0 <= --relation-start < --relation-full <= 1')
    main(args)
