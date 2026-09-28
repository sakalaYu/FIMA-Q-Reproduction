"""CPU tests for cached Attention/MLP suffixes, hooks, and mode restoration."""
import unittest
import torch
from torch import nn
from utils.fisher_probe_model import capture_branch, make_suffix


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3,3)
        self.mode = 'raw'

    def forward(self,x):
        y = self.layer(x).tanh()
        return y if self.mode=='raw' else y*0.9


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1,self.norm2 = nn.LayerNorm(3),nn.LayerNorm(3)
        self.attn,self.mlp = Branch(),Branch()
        self.ls1,self.ls2,self.drop_path1,self.drop_path2 = [nn.Identity() for _ in range(4)]

    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        return x+self.mlp(self.norm2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([Block(),Block()])
        self.norm=nn.LayerNorm(3)
        self.head=nn.Linear(3,2)

    def forward_head(self,x): return self.head(x.mean(1))

    def forward(self,x):
        for block in self.blocks: x=block(x)
        return self.forward_head(self.norm(x))


class SuffixTests(unittest.TestCase):
    def test_suffix_matches_full_model_and_remains_differentiable(self):
        model=Model().eval().requires_grad_(False)
        image=torch.randn(1,4,3)
        for index in (0,1):
            for branch in ('attn','mlp'):
                x,h,error,logits=capture_branch(model,index,branch,image)
                fn=make_suffix(model,index,branch,x)
                torch.testing.assert_close(fn(h),logits)
                self.assertGreater(float(error.norm()),0)
                self.assertEqual(getattr(model.blocks[index],branch).mode,'raw')
                h=h.requires_grad_()
                self.assertIsNotNone(torch.autograd.grad(fn(h).sum(),h)[0])
                self.assertFalse(model.blocks[index]._forward_pre_hooks)
                self.assertFalse(getattr(model.blocks[index],branch)._forward_hooks)


if __name__ == '__main__': unittest.main()
