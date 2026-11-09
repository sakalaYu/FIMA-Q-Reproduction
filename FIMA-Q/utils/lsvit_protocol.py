"""Explicit control recipes for the existing streaming LS-ViT adaptation."""


def validate_lsvit_control(args):
    control = args.lsvit_control
    if args.method_adaptation != 'lsvit':
        if control != 'c':
            raise ValueError('--lsvit-control requires --method-adaptation lsvit')
        return
    if not args.optimize or args.load_calibrate_checkpoint or args.load_optimize_checkpoint:
        raise ValueError('LS controls must reconstruct freshly from the FP model')
    if (args.endpoint_task != 'none' or args.endpoint_mode != 'none' or args.qwt
            or args.probe_kd != 'none' or args.attention_relation != 'none'):
        raise ValueError('LS controls exclude endpoint, QwT, probe KD and attention auxiliaries')
    expected_variant = {'plain': 'none', 'fixed': 'none', 'b': 'B', 'c': 'C'}[control]
    expected_rotation = control != 'plain'
    if (args.qk_joint_variant != expected_variant or args.headwise_hadamard != expected_rotation
            or args.headwise_mode != 'all'):
        raise ValueError(f'LS control {control} requires rotation={expected_rotation}, QK={expected_variant}')
    if expected_variant != 'none' and args.qk_joint_reconstruction_loss != 'mse':
        raise ValueError('B/C LS controls require explicit MSE pre-calibration')
