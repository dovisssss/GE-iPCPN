import torch
from einops import repeat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def integration_operator(batch, seq_len, dt):
    phi_1 = torch.tril(torch.ones(seq_len, seq_len))
    phi_1[0,0] = 0.0

    phi_2 = torch.tril(torch.ones(seq_len, seq_len))
    phi_2[:,0] = 0.0
    phi_2.fill_diagonal_(0.0)

    phi = (phi_1 + phi_2) * dt / 2
    Phi = repeat(phi, "row col -> batch row col", batch=batch)

    return Phi

def differentiation_operator(batch, seq_len, dt):
    phi_1_part1 = torch.tensor([[-3/2, 2, -1/2]], dtype=torch.float32)
    phi_1_part2 = torch.zeros(1, seq_len - 3)
    phi_1 = torch.cat([phi_1_part1, phi_1_part2], dim=1)

    temp_1 = -0.5 * torch.eye(seq_len - 2, dtype=torch.float32)
    temp1_pad = torch.cat([temp_1, torch.zeros(seq_len - 2, 2)], dim=1)
    temp_2 = 0.5 * torch.eye(seq_len - 2, dtype=torch.float32)
    temp2_pad = torch.cat([torch.zeros(seq_len - 2, 2), temp_2], dim=1)
    phi_2 = temp1_pad + temp2_pad

    phi_3_part1 = torch.zeros(1, seq_len - 3)
    phi_3_part2 = torch.tensor([[-1/2, 2, -3/2]], dtype=torch.float32)
    phi_3 = torch.cat([phi_3_part1, phi_3_part2], dim=1)

    phi = torch.cat([phi_1, phi_2, phi_3], dim=0) #按行拼接dim=0
    phi = (1 / dt) * phi
    Phi = repeat(phi, "row col -> batch row col", batch=batch)

    return Phi

#可能的问题：张量在GPU进行创建，后续调用方可能需要先将其移至cpu