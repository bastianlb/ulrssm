import os, re
import numpy as np
import scipy.io as sio
from itertools import product
from glob import glob
from pathlib import Path

import torch
# from torch.utils.data import Dataset

from ..utils.shape_util import read_shape
from ..utils.geometry_util import get_operators
from ..utils.registry import DATASET_REGISTRY

import potpourri3d as pp3d

from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import KNNGraph


def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', str(x)).group())))
    except AttributeError:
        return sorted(l)


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], None,
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)  # 在不同的阶段读取不同的东西
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L.to_dense()

    return item


class SingleShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=False,
                 return_evecs=True, num_evecs=200,
                 return_corr=False, return_dist=False, return_gl=False,
                 gl_feature_path="graph_laplacian", sample_and_indices=None,
                 return_dino=False, dino_feature_path="dino_features", graph_data=False):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
            return_corr (bool, optional): Indicate whether return the correspondences to reference shape. Default True.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.return_gl = return_gl
        self.return_dino = return_dino
        self.num_evecs = num_evecs
        self.sample_and_indices = sample_and_indices
        self.graph_data = graph_data

        if self.return_gl:
            self.gl_path = gl_feature_path
        if self.return_dino:
            self.dino_path = dino_feature_path

        self.off_files = []
        self.gl_feature_files = [] if self.return_gl else None
        self.dino_feature_files = [] if self.return_dino else None
        self.corr_files = [] if self.return_corr else None
        self.dist_files = [] if self.return_dist else None

        if self.sample_and_indices and self.sample_and_indices.get('sampled') is not None:
            self.sampled = self.sample_and_indices['sampled'] 
            # print('sampled!!!!', self.sampled)
            self.index_path = self.sample_and_indices['indices_relative_path'] if self.sample_and_indices.get('indices_relative_path') else None
        else:
            self.sampled = None
            self.index_path = []

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0
        if self.return_gl:
            assert self._size == len(self.gl_feature_files)

        if self.return_dist:
            assert self._size == len(self.dist_files)

        if self.return_corr:
            assert self._size == len(self.corr_files)
        
        self.graph_former = KNNGraph(k=10)

    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        
        self.off_files = sort_list(glob(f'{off_path}/*'))  # 获取并排序所有文件
        self.off_files = [f for f in self.off_files if f.endswith('.off') or f.endswith('.ply')]  # 过滤出.off和.ply文件

        if self.return_gl:
            gl_feature_path = Path(self.data_root) / self.gl_path
            self.gl_feature_files = sort_list(
                gl_feature_path.glob("*.pt"),
            )
            assert gl_feature_path.exists() and len(self.gl_feature_files) > 0, f'Invalid path {gl_feature_path} does not contain any .pt files'

        if self.return_dino:
            dino_feature_path = Path(self.data_root) / self.dino_path
            self.dino_feature_files = sort_list(
                dino_feature_path.glob("*.pt"),
            )
            assert dino_feature_path.exists() and len(self.dino_feature_files) > 0, f'Invalid path {dino_feature_path} does not contain any .pt files'

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(Path(self.data_root).parent, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))

        if self.sampled:
            self.index_path = os.path.join(self.data_root, self.index_path)
            if os.path.exists(self.index_path):
                file_extension = os.path.splitext(self.index_path)[1]
                if file_extension == '.npy':
                    self.index = torch.from_numpy(np.load(self.index_path))
                elif file_extension == '.pt':
                    self.index = torch.load(self.index_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
            else:
                raise FileNotFoundError(f"The file {self.index_path} does not exist.")

    def __getitem__(self, index):
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        # verts, faces = read_shape(off_file)
        verts, faces = pp3d.read_mesh(off_file)
        item['verts'] = torch.from_numpy(np.ascontiguousarray(verts)).float()
        
        if self.graph_data:
            item['data'] = Data(pos=item['verts'], x=item['verts'])
            item['graph'] = self.create_graph(item['verts'])
        
        if self.return_faces:
            item['faces'] = torch.from_numpy(np.ascontiguousarray(faces)).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=os.path.join(self.data_root, 'diffusion')) 

        if self.return_gl:
            gl_feature_file = self.gl_feature_files[index]
            # assert Path(off_file).name in gl_feature_file.name, f"The mesh file {Path(off_file).name} does not match with the gl_feature file {gl_feature_file.name}"
            graph_data = torch.load(gl_feature_file)
            item['gl_evecs'] = torch.tensor(graph_data['evecs'])
            item['gl_eval'] = torch.tensor(graph_data['evals'])
            assert item['gl_evecs'].shape[0] == item['verts'].shape[0]

        if self.return_dino:
            dino_feature_file = self.dino_feature_files[index]
            assert Path(off_file).name in dino_feature_file.name, f"The mesh file {Path(off_file).name} does not match with the dino feature file {dino_feature_file.name}"
            dino_data = torch.load(dino_feature_file)
            item['dino_features'] = dino_data  # already a tensor
            assert item['dino_features'].shape[0] == item['verts'].shape[0]

        # get geodesic distance matrix
        if self.return_dist:
            dist_file = self.dist_files[index]
            assert Path(off_file).stem in Path(dist_file).name, f"The mesh file {Path(off_file).name} does not match with the dist file {dist_file}"
            mat = sio.loadmat(dist_file)
            item['dist'] = torch.from_numpy(mat['dist']).float()
            # We have sampled on the preprocess.py 
            if self.sampled:
                item['dist'] = item['dist'][self.index, :][:, self.index] #TODO:Here we still use mesh version to calculate the dist(also possible to change to pcd, but currently using mesh to calculate this)
                # print('Sampled distance mat shape', item['dist'].shape)
        # get correspondences
        if self.return_corr:
            corr_file = self.corr_files[index]
            assert Path(off_file).stem in Path(corr_file).name, f"The mesh file {Path(off_file).name} does not match with the corr file {corr_file}"
            corr = np.loadtxt(corr_file, dtype=np.int32) - 1  # minus 1 to start from 0
        else:
            corr = np.arange(item['verts'].shape[0])
        item['corr'] = torch.from_numpy(corr).long()
        # print(item['corr'].shape)

        return item

    def __len__(self):
        return self._size
    
    def create_graph(self, vertices: torch.Tensor) -> Data:
        """
        Creates a graph from a given input point cloud through kNN.
        Args:
            vertices:       input point cloud
        Returns:
            graph:          output graph
        """
        graph = Data(pos=vertices, x=vertices)
        graph = self.graph_former(graph)
        edges = graph.edge_index
        edge_attr = torch.ones(edges.shape[1], dtype=torch.float32)
        graph.edge_attr = edge_attr

        return graph


@DATASET_REGISTRY.register()
class SingleFaustDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, *args, **kwds):
        super(SingleFaustDataset, self).__init__(data_root, *args, **kwds)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
            if self.dist_files:
                self.dist_files = self.dist_files[:80]
            if self.gl_feature_files:
                self.gl_feature_files = self.gl_feature_files[:80]
            if self.dino_feature_files:
                self.dino_feature_files = self.dino_feature_files[:80]
            self._size = 80
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
            if self.dist_files:
                self.dist_files = self.dist_files[80:]
            if self.gl_feature_files:
                self.gl_feature_files = self.gl_feature_files[80:]
            if self.dino_feature_files:
                self.dino_feature_files = self.dino_feature_files[80:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleScapeDataset(SingleShapeDataset):
    def __init__(self, *args, phase, **kwds):
        super(SingleScapeDataset, self).__init__(*args, **kwds)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 71, f'FAUST dataset should contain 71 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:51]
            if self.corr_files:
                self.corr_files = self.corr_files[:51]
            if self.dist_files:
                self.dist_files = self.dist_files[:51]
            self._size = 51
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[51:]
            if self.corr_files:
                self.corr_files = self.corr_files[51:]
            if self.dist_files:
                self.dist_files = self.dist_files[51:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleShrec19Dataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_dist=False, return_gl=False, sample_and_indices=None):
        super(SingleShrec19Dataset, self).__init__(data_root, return_faces, return_evecs, num_evecs, False, return_dist, return_gl, sample_and_indices)


@DATASET_REGISTRY.register()
class SingleSmalDataset(SingleShapeDataset):
    def __init__(self, *args, phase='train', category=True, **kwds):
        '''
        TODO: should also modify this part as before for new input variables
        '''
        
        self.phase = phase
        self.category = category
        super(SingleSmalDataset, self).__init__(*args, **kwds)

    def _init_data(self):
        if self.category:
            txt_file = os.path.join(self.data_root, f'{self.phase}_cat.txt')
        else:
            txt_file = os.path.join(self.data_root, f'{self.phase}.txt')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                if self.return_corr:
                    self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                if self.return_dist:
                    self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]
                # TODO: i don't like hardcoding these here again, but we dont need to rewrite his dataloading
                if self.return_gl:
                    self.gl_feature_files += [Path(self.data_root) / self.gl_path / f'{line}.off.pt']
                if self.return_dino:
                    self.dino_feature_files += [Path(self.data_root) / self.dino_path / f'{line}.off.pt']


@DATASET_REGISTRY.register()
class SingleDT4DDataset(SingleShapeDataset):
    def __init__(self, *args, phase='train', **kwds):
        self.phase = phase
        self.ignored_categories = ['pumpkinhulk']
        super(SingleDT4DDataset, self).__init__(*args, **kwds)

    def _init_data(self):
        with open(os.path.join(self.data_root, f'{self.phase}.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.split('/')[0] not in self.ignored_categories:
                    self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                    if self.return_corr:
                        self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                    if self.return_dist:
                        self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]
                    # TODO: i don't like hardcoding these here again, but we dont need to rewrite his dataloading
                    if self.return_gl:
                        self.gl_feature_files += [Path(self.data_root) / self.gl_path / f'{line}.off.pt']
                    if self.return_dino:
                        self.dino_feature_files += [Path(self.data_root) / self.dino_path / f'{line}.off.pt']


@DATASET_REGISTRY.register()
class SingleShrec20Dataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200, return_gl=False, sample_and_indices=None):
        super(SingleShrec20Dataset, self).__init__(data_root, return_faces,
                                                   return_evecs, num_evecs, False, False, return_gl, sample_and_indices)


@DATASET_REGISTRY.register()
class SingleTopKidsDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200, return_dist=False, return_gl=False, sample_and_indices=None):
        super(SingleTopKidsDataset, self).__init__(data_root, return_faces,
                                                   return_evecs, num_evecs, False, return_dist, return_gl, sample_and_indices)


class PairShapeDataset(Dataset):
    def __init__(self, dataset):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        self.combinations = list(product(range(len(dataset)), repeat=2))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]
        # print("processing shapes:", item['first']['name'], item['second']['name'])

        return item

    def __len__(self):
        return len(self.combinations)


@DATASET_REGISTRY.register()
class PairDataset(PairShapeDataset):
    def __init__(self, *args, **kwds):
        dataset = SingleShapeDataset(*args, **kwds)
        super(PairDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairFaustDataset(PairShapeDataset):
    def __init__(self, *args, **kwds):
        dataset = SingleFaustDataset(*args, **kwds)
        super(PairFaustDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairScapeDataset(PairShapeDataset):
    def __init__(self, *args, **kwds):
        dataset = SingleScapeDataset(*args, **kwds)
        super(PairScapeDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairShrec19Dataset(Dataset):
    def __init__(self, data_root, phase='test',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_dist=False, return_gl=True, gl_feature_path=None, sample_and_indices=None):
        assert phase in ['train', 'test'], f'Invalid phase: {phase}'
        self.dataset = SingleShrec19Dataset(data_root, return_faces, return_evecs, num_evecs, return_dist, return_gl, gl_feature_path, sample_and_indices)
        self.phase = phase
        if phase == 'test':
            corr_path = os.path.join(data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            # ignore the shape 40, since it is a partial shape
            self.corr_files = list(filter(lambda x: '40' not in x, sort_list(glob(f'{corr_path}/*.map'))))
            self._size = len(self.corr_files)
        else:
            self.combinations = list(product(range(len(self.dataset)), repeat=2))
            self._size = len(self.combinations)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.phase == 'train':
            # get index
            first_index, second_index = self.combinations[index]
        else:
            # extract pair index
            basename = os.path.basename(self.corr_files[index])
            indices = os.path.splitext(basename)[0].split('_')
            first_index = int(indices[0]) - 1
            second_index = int(indices[1]) - 1

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        if self.phase == 'test':
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['first']['corr'] = torch.arange(0, len(corr)).long()
            item['second']['corr'] = torch.from_numpy(corr).long()
        return item


@DATASET_REGISTRY.register()
class PairSmalDataset(PairShapeDataset):
    def __init__(self, *args, phase='train', **kwds):
        dataset = SingleSmalDataset(*args, phase=phase, **kwds)
        super(PairSmalDataset, self).__init__(dataset=dataset)


@DATASET_REGISTRY.register()
class PairDT4DDataset(PairShapeDataset):
    def __init__(self, *args, phase='train',
                 inter_class=False, **kwds):
        dataset = SingleDT4DDataset(*args, phase=phase, **kwds)
        super(PairDT4DDataset, self).__init__(dataset=dataset)
        self.inter_class = inter_class
        self.combinations = []
        if self.inter_class:
            self.inter_cats = set()
            files = os.listdir(os.path.join(self.dataset.data_root, 'corres', 'cross_category_corres'))
            for file in files:
                cat1, cat2 = os.path.splitext(file)[0].split('_')
                self.inter_cats.add((cat1, cat2))
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                # same category
                cat1, cat2 = self.dataset.off_files[i].split('/')[-2], self.dataset.off_files[j].split('/')[-2]
                if cat1 == cat2:
                    if not self.inter_class:
                        self.combinations.append((i, j))
                else:
                    if self.inter_class and (cat1, cat2) in self.inter_cats:
                        self.combinations.append((i, j))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]
        if self.dataset.return_corr and self.inter_class:
            # read inter-class correspondence
            first_cat = self.dataset.off_files[first_index].split('/')[-2]
            second_cat = self.dataset.off_files[second_index].split('/')[-2]
            corr = np.loadtxt(os.path.join(self.dataset.data_root, 'corres', 'cross_category_corres',
                                           f'{first_cat}_{second_cat}.vts'), dtype=np.int32) - 1
            item['second']['corr'] = item['second']['corr'][corr]

        return item


@DATASET_REGISTRY.register()
class PairShrec20Dataset(PairShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=120, return_gl=True, gl_feature_path=None, sample_and_indices=None):
        dataset = SingleShrec20Dataset(data_root, return_faces, return_evecs, num_evecs, return_gl, gl_feature_path, sample_and_indices)
        super(PairShrec20Dataset, self).__init__(dataset=dataset)


@DATASET_REGISTRY.register()
class PairShrec16Dataset(Dataset):
    """
    Pair SHREC16 Dataset
    """
    categories = [
        'cat', 'centaur', 'david', 'dog', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(self,
                 data_root,
                 categories=None,
                 cut_type='cuts', return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=False, return_dist=False, return_gl=True, gl_feature_path=None, sample_and_indices=None):
        assert cut_type in ['cuts', 'holes'], f'Unrecognized cut type: {cut_type}'

        categories = self.categories if categories is None else categories
        # sanity check
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = sorted(categories)
        self.cut_type = cut_type

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs

        # full shape files
        self.full_off_files = dict()
        self.full_dist_files = dict()

        # partial shape files
        self.partial_off_files = dict()
        self.partial_corr_files = dict()

        # load full shape files
        off_path = os.path.join(data_root, 'null', 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files'
        for cat in self.categories:
            off_file = os.path.join(off_path, f'{cat}.off')
            assert os.path.isfile(off_file)
            self.full_off_files[cat] = off_file

        if return_dist:
            dist_path = os.path.join(data_root, 'null', 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} without .mat files'
            for cat in self.categories:
                dist_file = os.path.join(dist_path, f'{cat}.mat')
                assert os.path.isfile(dist_file)
                self.full_dist_files[cat] = dist_file

        # load partial shape files
        self._size = 0
        off_path = os.path.join(data_root, cut_type, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files.'
        for cat in self.categories:
            partial_off_files = sorted(glob(os.path.join(off_path, f'*{cat}*.off')))
            assert len(partial_off_files) != 0
            self.partial_off_files[cat] = partial_off_files
            self._size += len(partial_off_files)

        if self.return_corr:
            # check the data path contains .vts files
            corr_path = os.path.join(data_root, cut_type, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} without .vts files.'
            for cat in self.categories:
                partial_corr_files = sorted(glob(os.path.join(corr_path, f'*{cat}*.vts')))
                assert len(partial_corr_files) == len(self.partial_off_files[cat])
                self.partial_corr_files[cat] = partial_corr_files

    def _get_category(self, index):
        assert index < len(self)
        size = 0
        for cat in self.categories:
            if index < size + len(self.partial_off_files[cat]):
                return cat, index - size
            else:
                size += len(self.partial_off_files[cat])

    def __getitem__(self, index):
        # get category
        cat, index = self._get_category(index)

        # get full shape
        full_data = dict()
        # get vertices
        off_file = self.full_off_files[cat]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        full_data['name'] = basename
        verts, faces = read_shape(off_file)
        full_data['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            full_data['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            full_data = get_spectral_ops(full_data, self.num_evecs, cache_dir=os.path.join(self.data_root, 'null',
                                                                                           'diffusion')) #need to change here!!!

        # get geodesic distance matrix
        if self.return_dist:
            dist_file = self.full_dist_files[cat]
            mat = sio.loadmat(dist_file)
            full_data['dist'] = torch.from_numpy(mat['dist']).float()

        # get partial shape
        partial_data = dict()
        # get vertices
        off_file = self.partial_off_files[cat][index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data['name'] = basename
        verts, faces = read_shape(off_file)
        partial_data['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            partial_data['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            partial_data = get_spectral_ops(partial_data, self.num_evecs,
                                            cache_dir=os.path.join(self.data_root, self.cut_type, 'diffusion'))

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.partial_corr_files[cat][index], dtype=np.int32) - 1
            full_data['corr'] = torch.from_numpy(corr).long()
            partial_data['corr'] = torch.arange(0, len(corr)).long()

        return {'first': full_data, 'second': partial_data}

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class PairTopKidsDataset(Dataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_dist=False, return_gl=True, gl_feature_path=None, sample_and_indices=None):
        assert phase in ['train', 'test'], f'Invalid phase: {phase}'
        self.dataset = SingleTopKidsDataset(data_root, return_faces, return_evecs, num_evecs, return_dist, return_gl, gl_feature_path, sample_and_indices)
        self.phase = phase
        if phase == 'test':
            corr_path = os.path.join(data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))
            self._size = len(self.corr_files)
        else:
            self.combinations = list(product(range(len(self.dataset)), repeat=2))
            self._size = len(self.combinations)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.phase == 'train':
            # get index
            first_index, second_index = self.combinations[index]
        else:
            # extract pair index
            first_index, second_index = 0, index + 1

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        if self.phase == 'test':
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['first']['corr'] = torch.from_numpy(corr).long()
            item['second']['corr'] = torch.arange(0, len(corr)).long()

        return item
