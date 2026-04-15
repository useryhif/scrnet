import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
def elements2vector(thing):
    elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    element = [str(i) for i in thing]
    ele_v=[0]*len(elements)
    for i in element:
        index = elements.index(i)
        ele_v[index] = 1
    return ele_v

class XRDDataset(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir,cls=None):
        self.root_dir = root_dir
        self.data=[]
        self.labels=[]
        cls_map = {"crystal_system":0,
                   "lattice": 1,
                   "point_group":2,
                   "spacegroup":3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P","monoclinic_P","monoclinic_C","orthorhombic_P","orthorhombic_C","orthorhombic_F","orthorhombic_I","tetragonal_P","tetragonal_I","trigonal_R","hexagonal_P","cubic_P","cubic_F","cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        sapce_group = [f"spacegroup{i+1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, sapce_group]
        cls_idx = {"crystal_system":2, "lattice":4, "point_group":5, "space_group":3}
        self.idx = cls_idx[cls]
        classes = li[cls_map[cls]]
        # print(classes)
        # classes = os.listdir("/home/light/PycharmProjects/material/resampled_test/ori_mutidata_cs_ori_gau1500")
        # classes = [i.split(".")[0] for i in os.listdir("/home/light/PycharmProjects/material/resampled_test/ori_mutidata_cs_ori_gau1500")]
        self.label_map={classes[i]:i for i in range(len(classes))}
        datas = np.load(root_dir, allow_pickle=True)#id y ele cslabel gslabell
        df = pd.DataFrame(datas)

        for sample in datas:
            xrd=sample
            # elements = one_class[3]
            # ele_vet = elements2vector(elements)
            # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
            self.data.append(xrd)
            self.labels.append(self.label_map[xrd[self.idx]])


    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map



    def __getitem__(self, idx):
        xrd = self.data[idx]
        data=torch.tensor(self.data[idx][-1], dtype=torch.float32)

        label=torch.tensor(self.labels[idx], dtype=torch.int64)
        return data,label, xrd #[1:]


class XRDDataset4resnet(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        sapce_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, sapce_group]
        cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = cls_idx[cls]
        classes = li[cls_map[cls]]
        # print(classes)
        # classes = os.listdir("/home/light/PycharmProjects/material/resampled_test/ori_mutidata_cs_ori_gau1500")
        # classes = [i.split(".")[0] for i in os.listdir("/home/light/PycharmProjects/material/resampled_test/ori_mutidata_cs_ori_gau1500")]
        self.label_map = {classes[i]: i for i in range(len(classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell

        # if drop_list is None:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 50].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)

        df = pd.DataFrame(self.datas)

        for sample in self.datas:
            xrd = sample
            # elements = one_class[3]
            # ele_vet = elements2vector(elements)
            # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
            self.data.append(xrd)
            self.labels.append(self.label_map[xrd[self.idx]])

    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def get_droplist(self):
        return self.low_count_classes

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][-1], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, xrd  # [1:]

class muti_XRDDataset4resnet(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir,cls=None,  drop_list=None):
        self.root_dir = root_dir
        self.data=[]
        self.labels=[]
        self.cls = cls
        cls_map = {"crystal_system":0,
                   "lattice": 1,
                   "point_group":2,
                   "space_group":3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P","monoclinic_P","monoclinic_C","orthorhombic_P","orthorhombic_C","orthorhombic_F","orthorhombic_I","tetragonal_P","tetragonal_I","trigonal_R","hexagonal_P","cubic_P","cubic_F","cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i+1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system":2, "lattice":4, "point_group":5, "space_group":3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map={self.classes[i]:i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)#id y ele cslabel gslabell

        # if drop_list is None:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 50].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)

        for sample in self.datas:
            # if 25<=self.label_map[sample[self.idx]]<32:
                xrd=sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])


    def __len__(self):
        return len(self.data)

    def get_droplist(self):
        return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.data)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data=torch.tensor(self.data[idx][6], dtype=torch.float32)
        pre=torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label=torch.tensor(self.labels[idx], dtype=torch.int64)
        return data,label, pre,xrd #[1:]

class muti_XRDDataset(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir,cls=None):
        self.root_dir = root_dir
        self.data=[]
        self.labels=[]
        self.cls = cls
        cls_map = {"crystal_system":0,
                   "lattice": 1,
                   "point_group":2,
                   "space_group":3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P","monoclinic_P","monoclinic_C","orthorhombic_P","orthorhombic_C","orthorhombic_F","orthorhombic_I","tetragonal_P","tetragonal_I","trigonal_R","hexagonal_P","cubic_P","cubic_F","cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i+1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system":2, "lattice":4, "point_group":5, "space_group":3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map={self.classes[i]:i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)#id y ele cslabel gslabell
        for sample in self.datas:
            # if 25<=self.label_map[sample[self.idx]]<32:
                xrd=sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])


    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.data)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data=torch.tensor(self.data[idx][6], dtype=torch.float32)
        pre=torch.tensor(self.data[idx][-2], dtype=torch.float32)
        label=torch.tensor(self.labels[idx], dtype=torch.int64)
        return data,label, pre,xrd #[1:]

class Sub_XRDDataset(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir,cls=None,start =0,length = 0,drop_list = None):
        self.root_dir = root_dir
        self.data=[]
        self.labels=[]
        self.cls = cls
        cls_map = {"crystal_system":0,
                   "lattice": 1,
                   "point_group":2,
                   "space_group":3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P","monoclinic_P","monoclinic_C","orthorhombic_P","orthorhombic_C","orthorhombic_F","orthorhombic_I","tetragonal_P","tetragonal_I","trigonal_R","hexagonal_P","cubic_P","cubic_F","cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i+1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system":2, "lattice":4, "point_group":5, "space_group":3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map={self.classes[i]:i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)#id y ele cslabel gslabell
        if drop_list is None:
            #去掉部分数据
            df = pd.DataFrame(self.datas)
            num_list = df.value_counts(self.idx)
            #得到所有数据小于n的类别
            self.low_count_classes = num_list[num_list < 100].index
            #获得所有标签为low_count_classes的索引
            indices = df[df[self.idx].isin(self.low_count_classes)].index
            #去掉这些数据
            self.datas = np.delete(self.datas, indices,0)
        else:
            # 去掉部分数据
            df = pd.DataFrame(self.datas)
            indices = df[df[self.idx].isin(drop_list)].index
            # 去掉这些数据
            self.datas = np.delete(self.datas, indices, 0)

            df = pd.DataFrame(self.datas)
            num_list = df.value_counts(self.idx)
            # 得到所有数据小于n的类别
            self.low_count_classes = num_list[num_list < 10].index
            # 获得所有标签为low_count_classes的索引
            indices = df[df[self.idx].isin(self.low_count_classes)].index
            # 去掉这些数据
            self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] <start + length:
                xrd=sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        indices = random.sample(range(len(self.others)), 3*len(self.data))
        selected_data = [self.others[i] for i in indices]
        selected_labels = [self.others_labels[i] for i in indices]
        selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        self.data.extend(selected_data)
        self.labels.extend(selected_labels)
        self.label_conf.extend(selected_labels_conf)
        if len(self.data)<256:
            self.data = self.data * (256 // len(self.data))*2
            self.labels = self.labels * (256 // len(self.labels))*2
            self.label_conf = self.label_conf * (256 // len(self.label_conf))*2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    def get_droplist(self):
        return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data=torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        pre=torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label=torch.tensor(self.labels[idx], dtype=torch.int64)
        return data,label,labels_conf, pre,xrd #[1:]

class Sub_XRDDataset4resnet(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        # indices = random.sample(range(len(self.others)), 3 * len(self.data))
        # selected_data = [self.others[i] for i in indices]
        # selected_labels = [self.others_labels[i] for i in indices]
        # selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        # self.data.extend(selected_data)
        # self.labels.extend(selected_labels)
        # self.label_conf.extend(selected_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        pre = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, labels_conf, pre, xrd  # [1:]

class Sub_XRDDataset_3attention(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        if len(self.others) >0:
            indices = random.sample(range(len(self.others)), len(self.data))
            selected_data = [self.others[i] for i in indices]
            selected_labels = [self.others_labels[i] for i in indices]
            selected_labels_conf = [self.others_labels_conf[i] for i in indices]
            self.data.extend(selected_data)
            self.labels.extend(selected_labels)
            self.label_conf.extend(selected_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)



    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        cs = torch.tensor(self.data[idx][-3], dtype=torch.float32)
        lattice = torch.tensor(self.data[idx][-2], dtype=torch.float32)
        pg = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, labels_conf, cs, lattice, pg, xrd  # [1:]
class Sub_XRDDataset_3attention_test(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        # indices = random.sample(range(len(self.others)), len(self.data))
        # selected_data = [self.others[i] for i in indices]
        # selected_labels = [self.others_labels[i] for i in indices]
        # selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        # self.data.extend(selected_data)
        # self.labels.extend(selected_labels)
        # self.label_conf.extend(selected_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)



    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        cs = torch.tensor(self.data[idx][-3], dtype=torch.float32)
        lattice = torch.tensor(self.data[idx][-2], dtype=torch.float32)
        pg = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, labels_conf, cs, lattice, pg, xrd  # [1:]
class Sub_XRDDataset_3attention_low_acc_idx(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        low_acc_index = np.load("low_acc_idx.npy")
        for sample in self.datas:
            if np.isin(self.label_map[sample[self.idx]], low_acc_index) :
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)
        indices = random.sample(range(len(self.others)), len(self.data)*3)
        selected_data = [self.others[i] for i in indices]
        selected_labels = [self.others_labels[i] for i in indices]
        selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        self.data.extend(selected_data)
        self.labels.extend(selected_labels)
        self.label_conf.extend(selected_labels_conf)
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        cs = torch.tensor(self.data[idx][-3], dtype=torch.float32)
        lattice = torch.tensor(self.data[idx][-2], dtype=torch.float32)
        pg = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, labels_conf, cs, lattice, pg, xrd  # [1:]
class Sub_XRDDataset_3attention_angle_abc(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        indices = random.sample(range(len(self.others)), len(self.data))
        selected_data = [self.others[i] for i in indices]
        selected_labels = [self.others_labels[i] for i in indices]
        selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        self.data.extend(selected_data)
        self.labels.extend(selected_labels)
        self.label_conf.extend(selected_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        cs = torch.tensor(self.data[idx][-5], dtype=torch.float32)
        lattice = torch.tensor(self.data[idx][-4], dtype=torch.float32)
        pg = torch.tensor(self.data[idx][-3], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        abc = torch.tensor(self.data[idx][-2], dtype=torch.float32)
        angles = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        return data, label, labels_conf, cs, lattice, pg, xrd,abc, angles  # [1:]
class Sub_XRDDataset_3attention4test(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        # indices = random.sample(range(len(self.others)), 3 * len(self.data))
        # selected_data = [self.others[i] for i in indices]
        # selected_labels = [self.others_labels[i] for i in indices]
        # selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        # self.data.extend(selected_data)
        # self.labels.extend(selected_labels)
        # self.label_conf.extend(selected_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        cs = torch.tensor(self.data[idx][-3], dtype=torch.float32)
        lattice = torch.tensor(self.data[idx][-2], dtype=torch.float32)
        pg = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, labels_conf, cs, lattice, pg, xrd  # [1:]

class Sub_XRDDataset_3attention4test_angle_abc(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0, drop_list=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        # if drop_list is None:
        #     #去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     #得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 100].index
        #     #获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     #去掉这些数据
        #     self.datas = np.delete(self.datas, indices,0)
        # else:
        #     # 去掉部分数据
        #     df = pd.DataFrame(self.datas)
        #     indices = df[df[self.idx].isin(drop_list)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        #
        #     df = pd.DataFrame(self.datas)
        #     num_list = df.value_counts(self.idx)
        #     # 得到所有数据小于n的类别
        #     self.low_count_classes = num_list[num_list < 10].index
        #     # 获得所有标签为low_count_classes的索引
        #     indices = df[df[self.idx].isin(self.low_count_classes)].index
        #     # 去掉这些数据
        #     self.datas = np.delete(self.datas, indices, 0)
        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        # indices = random.sample(range(len(self.others)), 3 * len(self.data))
        # selected_data = [self.others[i] for i in indices]
        # selected_labels = [self.others_labels[i] for i in indices]
        # selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        # self.data.extend(selected_data)
        # self.labels.extend(selected_labels)
        # self.label_conf.extend(selected_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    # def get_droplist(self):
    #     return self.low_count_classes

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        cs = torch.tensor(self.data[idx][-5], dtype=torch.float32)
        lattice = torch.tensor(self.data[idx][-4], dtype=torch.float32)
        pg = torch.tensor(self.data[idx][-3], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        abc = torch.tensor(self.data[idx][-2], dtype=torch.float32)
        angles = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        return data, label, labels_conf, cs, lattice, pg, xrd, abc, angles  # [1:]

class Sub_XRDDataset_(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0,drop_list = None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell

        if drop_list is None:
            #去掉部分数据
            df = pd.DataFrame(self.datas)
            num_list = df.value_counts(self.idx)
            #得到所有数据小于n的类别
            self.low_count_classes = num_list[num_list < 100].index
            #获得所有标签为low_count_classes的索引
            indices = df[df[self.idx].isin(self.low_count_classes)].index
            #去掉这些数据
            self.datas = np.delete(self.datas, indices,0)
        else:
            # 去掉部分数据
            df = pd.DataFrame(self.datas)
            indices = df[df[self.idx].isin(drop_list)].index
            # 去掉这些数据
            self.datas = np.delete(self.datas, indices, 0)

            df = pd.DataFrame(self.datas)
            num_list = df.value_counts(self.idx)
            # 得到所有数据小于n的类别
            self.low_count_classes = num_list[num_list < 10].index
            # 获得所有标签为low_count_classes的索引
            indices = df[df[self.idx].isin(self.low_count_classes)].index
            # 去掉这些数据
            self.datas = np.delete(self.datas, indices, 0)

        self.others = []
        self.others_labels = []
        self.label_conf = []
        self.others_labels_conf = []
        for sample in self.datas:
            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(self.label_map[xrd[self.idx]])
                self.label_conf.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(self.label_map[xrd[self.idx]])
                self.others_labels_conf.append(0)

        # indices = random.sample(range(len(self.others)), 3*len(self.data))
        # selected_data = [self.others[i] for i in indices]
        # selected_labels = [self.others_labels[i] for i in indices]
        # selected_labels_conf = [self.others_labels_conf[i] for i in indices]
        # self.data.extend(selected_data)
        # self.labels.extend(selected_labels)
        # self.label_conf.extend(selected_labels_conf)
        self.data.extend(self.others)
        self.labels.extend(self.others_labels)
        self.label_conf.extend(self.others_labels_conf)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
            self.label_conf = self.label_conf * (256 // len(self.label_conf)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def get_droplist(self):
        return self.low_count_classes

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        labels_conf = torch.tensor(self.label_conf[idx], dtype=torch.int64)
        pre = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, labels_conf, pre, xrd  # [1:]

class Sub_XRDDataset_conf(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None, start=0, length=0):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        self.others = []
        self.others_labels = []
        for sample in self.datas:

            if start <= self.label_map[sample[self.idx]] < start + length:
                xrd = sample
                self.data.append(xrd)
                self.labels.append(1)
            else:
                xrd = sample
                self.others.append(xrd)
                self.others_labels.append(0)
        indices = random.sample(range(len(self.others)),3*len(self.data))
        selected_data = [self.others[i] for i in indices]
        selected_labels = [self.others_labels[i] for i in indices]
        self.data.extend(selected_data)
        self.labels.extend(selected_labels)
        if len(self.data) < 256:
            self.data = self.data * (256 // len(self.data)) * 2
            self.labels = self.labels * (256 // len(self.labels)) * 2
        self.dataset = []
        for sample in self.datas:
            # if start <= self.label_map[sample[self.idx]] < start + length:
            self.dataset.append(sample)

    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.dataset)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        pre = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, pre, xrd  # [1:]

class Sub_XRDDataset_conf_test(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir, cls=None):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.cls = cls
        cls_map = {"crystal_system": 0,
                   "lattice": 1,
                   "point_group": 2,
                   "space_group": 3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P", "monoclinic_P", "monoclinic_C", "orthorhombic_P", "orthorhombic_C", "orthorhombic_F",
                   "orthorhombic_I", "tetragonal_P", "tetragonal_I", "trigonal_R", "hexagonal_P", "cubic_P", "cubic_F",
                   "cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm',
                                   '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm',
                                   '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i + 1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system": 2, "lattice": 4, "point_group": 5, "space_group": 3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map = {self.classes[i]: i for i in range(len(self.classes))}
        self.datas = np.load(root_dir, allow_pickle=True)  # id y ele cslabel gslabell
        self.others = []
        self.others_labels = []
        for sample in self.datas:
            xrd = sample
            self.data.append(xrd)
            self.labels.append(self.label_map[xrd[self.idx]])


    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.data)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.data[idx]
        data = torch.tensor(self.data[idx][6], dtype=torch.float32)
        pre = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return data, label, pre, xrd  # [1:]

class muti_XRDDataset4sg(Dataset):
    # def __init__(self, root_dir):
    #     self.root_dir = root_dir
    #     self.data=[]
    #     self.labels=[]
    #     classes=[name.split(".")[0] for name in os.listdir(root_dir)]
    #     self.label_map={classes[i]:i for i in range(len(classes))}
    #     for i in tqdm(os.listdir(root_dir)):
    #         path = os.path.join(root_dir, i)
    #         data=np.load(path, allow_pickle=True)
    #         for one_class in data:
    #             xrd=one_class[1]
    #             # elements = one_class[3]
    #             # ele_vet = elements2vector(elements)
    #             # xrd_ele = np.concatenate((xrd, ele_vet), axis=0)
    #             self.data.append(xrd)
    #             self.labels.append(self.label_map[i.split(".")[0]])
    def __init__(self, root_dir,cs_label,cls=None):
        self.root_dir = root_dir
        self.data=[]
        self.labels=[]
        self.cls = cls
        cls_map = {"crystal_system":0,
                   "lattice": 1,
                   "point_group":2,
                   "space_group":3,
                   }
        crystal_system = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
        lattice = ["triclinic_P","monoclinic_P","monoclinic_C","orthorhombic_P","orthorhombic_C","orthorhombic_F","orthorhombic_I","tetragonal_P","tetragonal_I","trigonal_R","hexagonal_P","cubic_P","cubic_F","cubic_I"]
        point_group = pg_labels = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
        space_group = [f"spacegroup{i+1}" for i in range(230)]
        li = [crystal_system, lattice, point_group, space_group]
        self.cls_idx = {"crystal_system":2, "lattice":4, "point_group":5, "space_group":3}
        self.idx = self.cls_idx[cls]
        self.classes = li[cls_map[cls]]
        # print(classes)
        self.label_map={self.classes[i]:i for i in range(len(self.classes))}
        classes_num = [2, 13, 59, 68, 25, 27, 36]
        acc_classes = [0,2,15,74,142,167,194]

        for i in range(len(acc_classes)):
            start = acc_classes[i]
            for j in range(classes_num[i]):
                self.label_map[f"spacegroup{start+1}"] = j
                start+=1
        self.datas = np.load(root_dir, allow_pickle=True)#id y ele cslabel gslabell
        self.cs_data = []
        self.cs_label = []
        for sample in self.datas:
            xrd=sample
            if xrd[2] ==cs_label:
                self.cs_data.append(xrd)
                self.cs_label.append(self.label_map[xrd[self.idx]])


    def __len__(self):
        return len(self.cs_data)

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        df = pd.DataFrame(self.cs_data)
        num_list = df.value_counts(self.idx).reindex(self.classes)
        return num_list

    def __getitem__(self, idx):
        xrd = self.cs_data[idx]
        data=torch.tensor(self.cs_data[idx][6], dtype=torch.float32)
        pre=torch.tensor(self.cs_data[idx][-1], dtype=torch.float32)
        label=torch.tensor(self.cs_label[idx], dtype=torch.int64)
        return data,label, pre,xrd #[1:]