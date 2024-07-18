import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CLLIE_BASISLoss(nn.Module):
    """
    
    """

    def __init__(self):
        """
        Init CLLIE_BASISLoss
        """
        super(CLLIE_BASISLoss, self).__init__()

    def forward(self, basis_A, basis_B, pc_B, correspondence12):
        '''
        pc_B here is the transposed version point
        correspondence12 here is a pcA * pcB shape matrix
        '''
        pseudo_inv_A = torch.pinverse(basis_A)
        C_opt = torch.matmul(pseudo_inv_A, correspondence12) @ basis_B
        opt_A = torch.matmul(basis_A, C_opt)

        # SoftMap
        dist_matrix = torch.cdist(opt_A, basis_B)       
        s_max = torch.nn.Softmax(dim=1)
        s_max_matrix = s_max(-dist_matrix)

        # Basis Loss
        losses = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.matmul(correspondence12, torch.transpose(pc_B,1,2))))
        
        return {'basis_loss': losses}

@LOSS_REGISTRY.register()
class CLLIE_DESCLoss(nn.Module):
    def __init__(self):
        """
        Init CLLIE_DESCLoss
        """
        super(CLLIE_DESCLoss, self).__init__()

    def forward(self, phi_A, phi_B, G_A, G_B):
        p_inv_phi_A = torch.pinverse(phi_A)
        p_inv_phi_B = torch.pinverse(phi_B)
        c_G_A = torch.matmul(p_inv_phi_A, G_A)
        c_G_B = torch.matmul(p_inv_phi_B, G_B)
        c_G_Bt = torch.transpose(c_G_B,2,1)

        # Estimated C
        C_my = torch.matmul(c_G_A,torch.transpose(torch.pinverse(c_G_Bt),2,1))

        # Optimal C
        C_opt = torch.matmul(p_inv_phi_A, phi_B)

        # MSE
        eucl_loss = torch.mean(torch.square(C_opt - C_my))

        return {'desc_loss': eucl_loss}

