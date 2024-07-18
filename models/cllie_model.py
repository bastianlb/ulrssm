import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
import time
import numpy as np
import scipy.sparse
import scipy.spatial
from sklearn.neighbors import NearestNeighbors

from utils.dist_util import master_only
from utils.tensor_util import to_numpy
from utils import get_root_logger
from collections import OrderedDict
import os

@MODEL_REGISTRY.register()
class CLLIEModel(BaseModel):
    '''
    Current version is purely same as the 20 paper, if we want to change the basis part, the left remains same, it is also still easily to be done.
    '''
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.stage = opt.get('stage', -1)
        
        if self.stage == -1:
            raise RuntimeError('we should give stage flag in the .yaml file')
        
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']

        super(CLLIEModel, self).__init__(opt)
        
        if self.stage == 2:
            load_last_stage_model = self.opt['path'].get('last_resume_state')
            if load_last_stage_model and os.path.isfile(load_last_stage_model):
                last_state_dict = torch.load(load_last_stage_model)
                self.load_last_stage_model(last_state_dict, net_only=True)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        correspondence_matrix = self.construct_corres(data_x['corr'], data_x['corr'], data_x['verts'], data_y['verts'])
        data_x['verts'] = data_x['verts'].transpose(2, 1)
        data_y['verts'] = data_y['verts'].transpose(2, 1)
        
        if self.stage == 1:
            basis_x, _, _ = self.networks['basis_encoder'](data_x['verts'])  # [B, Nx, C]
            basis_y, _, _ = self.networks['basis_encoder'](data_y['verts']) 
        
            self.loss_metrics = self.losses['CLLIE_BASISLoss'](basis_x, basis_y, data_y['verts'], correspondence_matrix)
        else:
            with torch.no_grad():
                self.networks['basis_encoder'].eval()
                basis_x, _, _ = self.networks['basis_encoder'](data_x['verts'])
                basis_y, _, _ = self.networks['basis_encoder'](data_y['verts'])

            desc_x, _, _ = self.networks['desc_encoder'](data_x['verts'])
            desc_y, _, _ = self.networks['desc_encoder'](data_y['verts'])
            self.loss_metrics = self.losses['CLLIE_DESCLoss'](basis_x, basis_y, desc_x, desc_y)               
                
    def construct_corres(self, vts_a, vts_b, verts_a, verts_b):
        """
        Compute the correspondence matrix out of two vts files
        Input: vts1: ndarray, shape: (bs, n1,)
            vts2: ndarray, shape: (bs,n2,)
        Output: Correspondence: ndarray, shape: (bs, n1, n2)
        """
        correspondence_matrix = torch.zeros((verts_a.shape[0], verts_a.shape[1], verts_b.shape[1]), device=verts_a.device)

        for i in range(vts_a.shape[1]):
            index_1 = int(vts_a[0, i])
            index_2 = int(vts_b[0, i])
            if index_1 < correspondence_matrix.shape[1] and index_2 < correspondence_matrix.shape[2]:
                correspondence_matrix[:, index_1, index_2] = 1
        correspondence_list = []
        correspondence_list_mask = torch.ones(correspondence_matrix.shape[0])
        
        # The following not used
        for j in range(correspondence_matrix.shape[1]):
            non_zero_counts = torch.sum(correspondence_matrix[:, j] == 1)
            if non_zero_counts > 0:
                correspondence_list.append(correspondence_matrix[:, j].nonzero()[0][0])
            else:
                correspondence_list.append(10000)
                correspondence_list_mask[j] = 0
        correspondence_list = torch.tensor(correspondence_list).detach().cpu().numpy().astype(int)
        correspondence_list_mask = correspondence_list_mask
        return correspondence_matrix
        
    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        num_a, num_b = data_x['verts'].shape[1], data_y['verts'].shape[1]
        data_x['verts'] = data_x['verts'].transpose(2, 1)
        data_y['verts'] = data_y['verts'].transpose(2, 1)
        
        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        with torch.no_grad():
            self.networks['basis_encoder'].eval()
            basis_x, _, _ = self.networks['basis_encoder'](data_x['verts'])
            basis_y, _, _ = self.networks['basis_encoder'](data_y['verts'])

            self.networks['desc_encoder'].eval()
            desc_x, _, _ = self.networks['desc_encoder'](data_x['verts'])
            desc_y, _, _ = self.networks['desc_encoder'](data_y['verts'])

        Pyx, Cxy, match = self.computeDeepCorrespondence(basis_x, basis_y, desc_x, desc_y, num_a, num_b)
        Pyx = Pyx.toarray()
        p2p = np.nonzero(Pyx)[1]

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy
    
    def computeDeepCorrespondence(self, basisA, basisB, descA, descB, num_vert_A, num_vert_B):
        basisA, basisB, descA, descB, num_vert_A, num_vert_B = to_numpy(basisA.squeeze()), to_numpy(basisB.squeeze()), to_numpy(descA.squeeze()), to_numpy(descB.squeeze()), to_numpy(num_vert_A).item(), to_numpy(num_vert_B).item()

        F = np.linalg.pinv(basisA).dot(descA)
        G = np.linalg.pinv(basisB).dot(descB)

        para = np.identity(basisA.shape[1])

        C = np.linalg.lstsq(np.kron(F.T, para), np.reshape(G, (-1), order='F'), rcond=None)[0]  # 最小二乘问题 C_init (25,)
        C = np.reshape(C, (basisA.shape[1], basisA.shape[1]), order='F')

        # C1 = np.linalg.lstsq(np.kron(G.T, para), np.reshape(F, (-1), order='F'), rcond=None)[0]  # 最小二乘问题 C_init (25,)
        # C1 = np.reshape(C, (self.basisA.shape[1], self.basisA.shape[1]), order='F')
        # C1 = F @ np.linalg.pinv(G)

        P = scipy.sparse.lil_matrix((num_vert_B, num_vert_A))  # （8018，5103）

        dataA = basisA
        dataB = basisB @ C

        dataTree = scipy.spatial.KDTree(dataB)

        dist, ind = dataTree.query(dataA, workers=6)  # dist是每个对应点之间的距离，ind是对应点索引，shape和A的点数一样

        P[ind, np.arange(num_vert_A)] = 1  # P:(8018,5103) ind是B的索引

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(basisB.dot(C.T))
        match = nn.kneighbors(basisA, n_neighbors=1, return_distance=False)[:, 0]

        return P, C, match
    
    def resume_model(self, resume_state, net_only=False, verbose=True):
        """Reload the net, optimizers and schedulers.

        Args:
            resume_state (dict): Resume state.
            net_only (bool): only resume the network state dict. Default False.
            verbose (bool): print the resuming process
        """
        networks_state_dict = resume_state['networks']

        # resume networks
        if self.stage == 1:
            for name in self.networks['basis_encoder']:
                if len(list(self.networks['basis_encoder'][name].parameters())) == 0:
                    if verbose:
                        logger = get_root_logger()
                        logger.info(f'Network {name} has no param. Ignore it.')
                    continue
                if name not in networks_state_dict:
                    if verbose:
                        logger = get_root_logger()
                        logger.warning(f'Network {name} cannot be resumed.')
                    continue

                net_state_dict = networks_state_dict[name]
                # remove unnecessary 'module.'
                net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

                self._get_bare_net(self.networks['basis_encoder'][name]).load_state_dict(net_state_dict)

                if verbose:
                    logger = get_root_logger()
                    logger.info(f"Resuming network: {name}")
        else:
            for name in self.networks['desc_encoder']:
                if len(list(self.networks['desc_encoder'][name].parameters())) == 0:
                    if verbose:
                        logger = get_root_logger()
                        logger.info(f'Network {name} has no param. Ignore it.')
                    continue
                if name not in networks_state_dict:
                    if verbose:
                        logger = get_root_logger()
                        logger.warning(f'Network {name} cannot be resumed.')
                    continue

                net_state_dict = networks_state_dict[name]
                # remove unnecessary 'module.'
                net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

                self._get_bare_net(self.networks['desc_encoder'][name]).load_state_dict(net_state_dict)

                if verbose:
                    logger = get_root_logger()
                    logger.info(f"Resuming network: {name}")

        # resume optimizers and schedulers
        if not net_only:
            optimizers_state_dict = resume_state['optimizers']
            schedulers_state_dict = resume_state['schedulers']
            for name in self.optimizers:
                if name not in optimizers_state_dict:
                    if verbose:
                        logger = get_root_logger()
                        logger.warning(f'Optimizer {name} cannot be resumed.')
                    continue
                self.optimizers[name].load_state_dict(optimizers_state_dict[name])
            for name in self.schedulers:
                if name not in schedulers_state_dict:
                    if verbose:
                        logger = get_root_logger()
                        logger.warning(f'Scheduler {name} cannot be resumed.')
                    continue
                self.schedulers[name].load_state_dict(schedulers_state_dict[name])

            # resume epoch and iter
            self.curr_iter = resume_state['iter']
            self.curr_epoch = resume_state['epoch']
            if verbose:
                logger = get_root_logger()
                logger.info(f"Resuming training from epoch: {self.curr_epoch}, " f"iter: {self.curr_iter}.")

    def load_last_stage_model(self, resume_state, verbose=True):
        networks_state_dict = resume_state['networks']
        for name in self.networks['basis_encoder']:
            net_state_dict = networks_state_dict[name]
            # remove unnecessary 'module.'
            net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

            network = self._get_bare_net(self.networks['basis_encoder'][name])
            network.load_state_dict(net_state_dict)

            # Freeze all parameters
            for param in network.parameters():
                param.requires_grad = False

            if verbose:
                logger = get_root_logger()
                logger.info(f"Resuming network: {name}")

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        if self.stage == 1: # we don't check the val for stage 1
            pass
        else:
            super(CLLIEModel, self).validation(dataloader, tb_logger, update)

