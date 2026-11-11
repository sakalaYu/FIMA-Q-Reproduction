"""Explicit control recipes for the existing streaming LS-ViT adaptation."""


def validate_lsvit_control(args):
    control = args.lsvit_control
    pair = getattr(args, 'ls_pair_dir', None)
    mr = getattr(args, 'ls_mr', 'none')
    if mr != 'none':
        screen = mr == 'screen'
        needs_directory = mr in ('screen', 'relu')
        if (pair or args.model != 'vit_small' or not args.calibrate or args.load_optimize_checkpoint
                or not args.expected_fp_model_sha256 or not args.headwise_hadamard or args.headwise_mode != 'all'
                or args.qk_joint_variant != 'none' or args.ls_mr_steps < 1
                or bool(args.ls_mr_dir) != needs_directory or args.optimize == screen
                or args.skip_final_validation != screen or (screen and args.method_adaptation != 'none')
                or (not screen and (args.method_adaptation != 'lsvit' or control != 'fixed'))):
            raise ValueError('MR requires fresh ViT-S with full Hadamard and fixed LS for quantization')
    if pair:
        prepare = (args.method_adaptation == 'none' and args.calibrate and not args.optimize
                   and not args.load_optimize_checkpoint and not args.load_calibrate_checkpoint)
        branch = (args.method_adaptation == 'lsvit' and control in ('mix-equal', 'mix-adaptive')
                  and args.optimize and args.load_calibrate_checkpoint and not args.load_optimize_checkpoint)
        if not (prepare or branch) or not args.expected_fp_model_sha256 or not args.device.startswith('cuda'):
            raise ValueError('Paired LS requires shared calibration or a mixture branch, CUDA, and the FP fingerprint')
        if prepare and (args.model != 'vit_small' or not args.headwise_hadamard or args.headwise_mode != 'all'
                        or not args.skip_final_validation or args.qk_joint_variant != 'none'):
            raise ValueError('Shared LS calibration requires ViT-S with fixed full Hadamard and calibration only')
    if args.method_adaptation != 'lsvit':
        if control != 'c':
            raise ValueError('--lsvit-control requires --method-adaptation lsvit')
        return
    if not args.optimize or (args.load_calibrate_checkpoint and not pair) or args.load_optimize_checkpoint:
        raise ValueError('LS controls must reconstruct freshly from the FP model')
    if (args.endpoint_task != 'none' or args.endpoint_mode != 'none' or args.qwt
            or args.probe_kd != 'none' or args.attention_relation != 'none'):
        raise ValueError('LS controls exclude endpoint, QwT, probe KD and attention auxiliaries')
    expected_variant = {'plain': 'none', 'fixed': 'none', 'b': 'B', 'c': 'C',
                        'mix-equal': 'none', 'mix-adaptive': 'none'}[control]
    expected_rotation = control != 'plain'
    if (args.qk_joint_variant != expected_variant or args.headwise_hadamard != expected_rotation
            or args.headwise_mode != 'all'):
        raise ValueError(f'LS control {control} requires rotation={expected_rotation}, QK={expected_variant}')
    if expected_variant != 'none' and args.qk_joint_reconstruction_loss != 'mse':
        raise ValueError('B/C LS controls require explicit MSE pre-calibration')
    if control.startswith('mix-') and (args.recon_iters != 20000 or args.optim_size != 1024
                                       or args.optim_mode != 'qdrop' or args.drop_prob != 0.5):
        raise ValueError('LS mixture controls require 20000 steps, 1024 images and QDrop 0.5')
