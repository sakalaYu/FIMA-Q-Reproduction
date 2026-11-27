"""Cache a standard pre-norm ViT branch context and replay only its FP suffix.

First implementation supports timm ViT/DeiT (not Swin/distilled variants).
Branch positions: attn after proj_drop, mlp after fc2/drop; before LayerScale
and residual addition. The caller verifies suffix == original FP logits.
"""
from contextlib import contextmanager
import torch


@contextmanager
def quant_mode(module, mode):
    saved = [(m, m.mode) for m in module.modules() if hasattr(m, 'mode')]
    try:
        for m, _ in saved:
            m.mode = mode
        yield
    finally:
        for m, old in saved:
            m.mode = old


def capture_branch(model, block_index, branch, image):
    block = model.blocks[block_index]
    target = getattr(block, branch)
    captured = {}
    hooks = [block.register_forward_pre_hook(lambda m, inputs: captured.update(x=inputs[0].detach())),
             target.register_forward_hook(lambda m, inputs, output:
                 captured.update(h=output.detach(), branch_input=inputs[0].detach()))]
    try:
        with torch.no_grad(), quant_mode(model, 'raw'):
            logits = model(image)
    finally:
        for hook in hooks:
            hook.remove()
    raw = captured['h']
    with torch.no_grad(), quant_mode(target, 'quant_forward'):
        quantized = target(captured['branch_input'])
    return captured['x'], raw, quantized - raw, logits.detach()


def make_suffix(model, block_index, branch, block_input):
    block = model.blocks[block_index]
    if branch not in ('attn', 'mlp'):
        raise ValueError('branch must be attn or mlp')
    if not all(hasattr(block, name) for name in ('ls1', 'ls2', 'drop_path1', 'drop_path2')):
        raise ValueError('Expected standard timm pre-norm ViT block')
    with torch.no_grad():
        residual = block_input.detach()
        if branch == 'mlp':
            residual = residual + block.drop_path1(block.ls1(block.attn(block.norm1(residual))))

    def suffix(h):
        if branch == 'attn':
            x = residual + block.drop_path1(block.ls1(h))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        else:
            x = residual + block.drop_path2(block.ls2(h))
        for later in model.blocks[block_index + 1:]:
            x = later(x)
        return model.forward_head(model.norm(x))
    return suffix
