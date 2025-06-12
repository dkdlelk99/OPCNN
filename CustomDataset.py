import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.preprocess_mol import smiles2FP, ValidSmiles
from utils.smiles2graph import smiles_to_graph



class FpDataset(Dataset):
    def __init__(self, data, data_name):
        super().__init__()
        self.data = data
        self.x = {"ecfp":[], "maccs": []}
        self.y = []
        self.smiles = []
        
        if data_name == "BBB":
            for mol in data.iterrows():
                if not ValidSmiles(mol[1]["Unnamed: 0"]):
                    continue
                self.smiles.append(mol[1]["Unnamed: 0"])
                FP = smiles2FP(mol[1]["Unnamed: 0"], "split_list")
                self.x["ecfp"].append(FP[0])
                self.x["maccs"].append(FP[1])
                self.y.append(mol[1]['BBclass'])
        elif data_name == "logP":
            for mol in data.iterrows():
                if not ValidSmiles(mol[1]["smiles"]):
                    continue
                self.smiles.append(mol[1]["smiles"])
                FP = smiles2FP(mol[1]["smiles"], "split_list")
                self.x["ecfp"].append(FP[0])
                self.x["maccs"].append(FP[1])
                self.y.append(mol[1]['logp'])
        else:
            print("data_name should be BBB or logP")
            return

        self.x['ecfp'] = torch.Tensor(self.x['ecfp'])
        self.x['maccs'] = torch.Tensor(self.x['maccs'])
        self.y = torch.Tensor(self.y)


    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return f"Fingerprint({len(self.y):,})"

    def __getitem__(self, idx):
        return {"ecfp": self.x['ecfp'][idx], "maccs": self.x['maccs'][idx], "y": self.y[idx]}
    



class GraphDataset(Dataset):
    def __init__(self, data, data_name):
        '''
        data: pandas DataFrame (smiles, target)
        '''
        super().__init__()
        self.data_list = []
        self.smiles = []
        self.y = []
        self.data_name = data_name

        for mol in data.iterrows():
            # 1. diff. data sources
            if data_name == "BBB":
                smiles = mol[1]["Unnamed: 0"]
                y = torch.Tensor([mol[1]['BBclass']])
            elif data_name == "logP":
                smiles = mol[1]["smiles"]
                y = torch.Tensor([mol[1]['logp']])
            else:
                print("data_name should be BBB or logP")
                return

            # 2. check valid smiles
            if not ValidSmiles(smiles):
                continue
            
            # 3. convert smiles to graph
            graph = smiles_to_graph(smiles)
            self.data_list.append(Data(
                x=graph['x'],
                edge_attr=graph['edge_attr'],
                edge_index=graph['edge_index'],
                y=y,
                smiles=smiles
            ))
            self.y.append(y)
            self.smiles.append(smiles)
        
        self.y = torch.Tensor(self.y)

        self.x = torch.cat([data.x for data in self.data_list])
        self.edge_attr = torch.cat([data.edge_attr for data in self.data_list])
        self.edge_index = torch.cat([data.edge_index for data in self.data_list], dim=1)


    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return f"{self.data_name} Graph({len(self.y):,})"

    def __getitem__(self, idx):
        return self.data_list[idx]
