import torch

class HardMoE(torch.nn.Module):
    def __init__(self, experts, gate):
        super().__init__()
        self.experts = torch.nn.ModuleList(experts)
        self.gate = gate

    def forward(self, x):
        gate_out = self.gate(x)
        gate_out_max_idxs = torch.argmax(gate_out, dim=1)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = expert_outs[torch.arange(expert_outs.size(0)), :, gate_out_max_idxs]
        return output