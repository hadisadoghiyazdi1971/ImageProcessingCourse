from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def dino_loss(student_logits, teacher_logits, student_temp=.1, teacher_temp=.04, center=None):
    if center is None:
        center = torch.zeros(1, teacher_logits.shape[-1], device=teacher_logits.device)
    p_t = F.softmax((teacher_logits - center) / teacher_temp, dim=-1).detach()
    log_p_s = F.log_softmax(student_logits / student_temp, dim=-1)
    return -(p_t * log_p_s).sum(dim=-1).mean()


class MLP(nn.Module):
    def __init__(self, d=64, out=256):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,out))
    def forward(self,x): return self.net(x)


if __name__ == '__main__':
    torch.manual_seed(5)
    student = MLP(); teacher = copy.deepcopy(student)
    for p in teacher.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), 1e-3)
    x1, x2 = torch.randn(32,64), torch.randn(32,64)
    s = student(x1)
    with torch.no_grad(): t = teacher(x2)
    loss=dino_loss(s,t)
    opt.zero_grad(); loss.backward(); opt.step(); update_teacher(student,teacher,.996)
    delta=sum((a-b).abs().mean() for a,b in zip(student.parameters(),teacher.parameters()))
    print({'loss':float(loss.detach()),'student_teacher_mean_delta':float(delta.detach())})
    assert torch.isfinite(loss)
