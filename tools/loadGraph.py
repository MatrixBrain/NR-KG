import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class Graph_data():
    def __init__(self, KGtype) -> None:
        self.KGtype = KGtype
        self.node_dict = {}
        self.edge_dict = {}
        self.node_name2id = {}
        self.edge_name2id = {}
        self.data_graphs = {}
        self.HEA_node_index = {}
        self.active_or_passive = {}  # 0:passive, 1:active

        self.max_y = {}
        self.min_y = {}
        self.max_hea = {}
        self.min_hea = {}

    def get_normalization_param(self, label):
        """
        Get the normalization parameters
        """
        param = {'max_hea': self.max_hea[label], 'min_hea': self.min_hea[label], 'max_y': self.max_y[label], 'min_y': self.min_y[label]}
        return param

    def get_graph(self, label):
        """
        Get the graph data
        """
        return self.data_graphs[label]

    def load_graph_from_neo4jcsv(self, addr, label='ori', node_type=0):
        """
        Load the graph data from the csv/pkl files 
        addr: the folder where the csv files are located
        label: the label of the graph
        """
        entity_df, rel_df = self.read_data(addr)
        self.node_dict[label], self.edge_dict[label], self.node_name2id[label], self.edge_name2id[label], data_graph, max_hea, min_hea, max_y, min_y, HEA_node_index = self.generate_graph(entity_df, rel_df, KGtype=self.KGtype)
        self.data_graphs[label] = data_graph.clone()
        self.active_or_passive[label] = 1
        self.max_hea[label] = max_hea.clone()
        self.min_hea[label] = min_hea.clone()
        self.max_y[label] = max_y.clone()
        self.min_y[label] = min_y.clone()
        self.HEA_node_index[label] = HEA_node_index
        return self.data_graphs[label]

    def add_graph_from_neo4jcsv(self, addr, label='add', base_label='ori', node_type=2):
        """
        Add the graph data from the csv/pkl files
        addr: the folder where the csv files are located
        label: the label of the graph
        base_label: the label of the base graph
        node_type: the type of the node 0: train 1: valid 2: test/predict
        """
        entity_df, rel_df = self.read_data(addr)
        self.node_dict[label], self.edge_dict[label], self.node_name2id[label], self.edge_name2id[label], data_graph, max_hea, min_hea, max_y, min_y, HEA_node_index = self.generate_graph(entity_df, rel_df, KGtype=self.KGtype, 
                                                                                                                                                               node_dict=self.node_dict[base_label], edge_dict=self.edge_dict[base_label], 
                                                                                                                                                               node_name2id=self.node_name2id[base_label], edge_name2id=self.edge_name2id[base_label], 
                                                                                                                                                               data_graph=self.data_graphs[base_label], 
                                                                                                                                                               node_type=node_type, 
                                                                                                                                                               max_hea=self.max_hea[base_label], min_hea=self.min_hea[base_label], max_y=self.max_y[base_label], min_y=self.min_y[base_label],
                                                                                                                                                               HEA_node_index_ori=self.HEA_node_index[base_label],
                                                                                                                                                               active_or_passive=self.active_or_passive[base_label])

        self.data_graphs[label] = data_graph.clone()
        self.active_or_passive[label] = self.active_or_passive[base_label]
        self.max_hea[label] = max_hea.clone()
        self.min_hea[label] = min_hea.clone()
        self.max_y[label] = max_y.clone()
        self.min_y[label] = min_y.clone()
        self.HEA_node_index[label] = HEA_node_index
        return self.data_graphs[label]    

    def element2weight(self, element):
        """
        Transfer the element to the edge weight
        """
        return element / 100 + 1
    
    def convert_passive_active(self, label, active_or_passive):
        """
        convert the direction of the relation
        """
        if type(active_or_passive) == type(""):
            if active_or_passive.lower() == "active":
                active_or_passive = 1
            elif active_or_passive.lower() == "passive":
                active_or_passive = 0

        if self.active_or_passive[label] == active_or_passive:
            return self.data_graphs[label]
        else:
            self.data_graphs[label] = self._convert_passive_active(self.data_graphs[label])
            self.active_or_passive[label] = active_or_passive
            return self.data_graphs[label]

    def read_data(self, addr):
        # file_list = os.listdir(addr)
        entity_df = pd.DataFrame()
        rel_df = pd.DataFrame()

        # for file in file_list:
        #     if file.startswith('entity_'):
        #         entity_df_ = pd.read_csv(os.path.join(addr, file))
        #         i = entity_df_.columns.str.contains(':ID')
        #         entity_df_.columns.values[i] = ':ID'
        #         entity_df = pd.concat([entity_df, entity_df_], axis=0)
        #     elif file.startswith('rel_'):
        #         rel_df = pd.concat([rel_df, pd.read_csv(os.path.join(addr, file))], axis=0)

        entity_df = pd.read_pickle(os.path.join(addr, 'entity.pkl'))
        rel_df = pd.read_pickle(os.path.join(addr, 'relation.pkl'))
        entity_df = entity_df.reset_index(drop=True)
        rel_df = rel_df.reset_index(drop=True)

        return entity_df, rel_df

    def generate_graph(self, entity_df, rel_df, KGtype, node_dict=None, edge_dict=None, node_name2id=None, edge_name2id=None, data_graph=None, node_type=0, max_hea=None, min_hea=None, max_y=None, min_y=None, HEA_node_index_ori=None, active_or_passive=None):
        """
        Generate the graph data
        """
        # 深拷贝
        if data_graph is not None:
            data_graph = data_graph.clone()
        if max_hea is not None:
            max_hea = max_hea.clone()
        if min_hea is not None:
            min_hea = min_hea.clone()
        if max_y is not None:
            max_y = max_y.clone()
        if min_y is not None:
            min_y = min_y.clone()

        if active_or_passive is not None:
            if active_or_passive == 0:
                data_graph = self._convert_passive_active(data_graph)


        D_y = 1
        if KGtype == 'HEA-HD-KG':
            D_y = 1
        elif KGtype == 'HEA-CRD-KG':
            D_y = 1

        if node_dict is None:
            node_id = 0
            edge_id = 0
            node_dict = {}
            edge_dict = {}
            node_name2id = {}
            edge_name2id = {}
        else:
            exist_node = len(node_dict.keys())
            node_id = len(node_dict.keys())
            edge_id = len(edge_dict.keys())

        HEA_node_index = []

        x = torch.zeros((entity_df.shape[0], 7))
        x_id = torch.zeros((entity_df.shape[0], 1))
        x_attr = torch.zeros((entity_df.shape[0], 1))
        x_type = torch.zeros((entity_df.shape[0], 1))
        y = torch.zeros((entity_df.shape[0], D_y))
        y_mask = torch.zeros((entity_df.shape[0], D_y))

        for i in range(entity_df.shape[0]):
            node = {}
            node['id'] = node_id
            node['node_id'] = entity_df.loc[i, ':ID']
            node['label'] = entity_df.loc[i, ':LABEL']
            if node['label'] == 'HEA':
                node['name'] = entity_df.loc[i, 'name']
            elif node['label'] == 'element':
                node['name'] = entity_df.loc[i, 'element']
            elif node['label'] == 'process_technology':
                node['name'] = entity_df.loc[i, 'process_technology']
            elif node['label'] == 'structure':
                node['name'] = entity_df.loc[i, 'structure']
            node_dict[node_id] = node
            node_id += 1
            node_name2id[node['node_id']] = node['id']

            if node['label'] == 'HEA':
                x[i] = torch.tensor([entity_df.loc[i, 'element_Al'], entity_df.loc[i, 'element_Fe'], entity_df.loc[i, 'element_Co'], entity_df.loc[i, 'element_Ni'], entity_df.loc[i, 'element_Cr'], entity_df.loc[i, 'element_Cu'], entity_df.loc[i, 'element_Mn']])
                x_id[i] = torch.tensor([node['id']])
                x_attr[i] = torch.tensor([1])
                HEA_node_index.append(i)
                x_type[i] = torch.tensor([node_type])
                if KGtype == 'HEA-HD-KG':
                    HV = np.nan
                    if not np.isnan(entity_df.loc[i, 'HV']):
                        HV = entity_df.loc[i, 'HV']
                    y_mask[i] = torch.tensor([0 if np.isnan(HV) else 1])
                    y[i] = torch.tensor([0 if np.isnan(HV) else HV])
                elif KGtype == 'HEA-CRD-KG':
                    corrosion = np.nan
                    if not np.isnan(entity_df.loc[i, 'Icorr(uA/cm2)']):
                        corrosion = np.log(entity_df.loc[i, 'Icorr(uA/cm2)'])
                    y_mask[i] = torch.tensor([0 if np.isnan(corrosion) else 1])
                    y[i] = torch.tensor([0 if np.isnan(corrosion) else corrosion])

            elif node['label'] == 'element' or node['label'] == 'process_technology' or node['label'] == 'structure':
                x[i] = torch.tensor([node['id']] * 7)
                x_id[i] = torch.tensor([node['id']])
                x_type[i] = torch.tensor([node_type])
                if node['label'] == 'element':
                    x_attr[i] = torch.tensor([2])
                elif node['label'] == 'process_technology':
                    x_attr[i] = torch.tensor([3])
                elif node['label'] == 'structure':
                    x_attr[i] = torch.tensor([4])

        edge_index = torch.zeros((2, rel_df.shape[0]), dtype=torch.long)
        edge_attr = torch.zeros((rel_df.shape[0], 1))
        edge_weight = torch.zeros((rel_df.shape[0], 1))

        for i in range(rel_df.shape[0]):
            if rel_df.loc[i, ':TYPE'] not in edge_dict.values():
                edge_dict[edge_id] = rel_df.loc[i, ':TYPE']
                edge_name2id[rel_df.loc[i, ':TYPE']] = edge_id
                edge_id += 1
            
            head_id = node_name2id[rel_df.loc[i, ':START_ID']]
            tail_id = node_name2id[rel_df.loc[i, ':END_ID']]
            edge_index[0, i] = head_id
            edge_index[1, i] = tail_id
            edge_attr[i] = edge_name2id[rel_df.loc[i, ':TYPE']]
            edge_weight[i] = 1
            if 'element' in rel_df.loc[i, ':END_ID']:
                if data_graph is not None:
                    head_id = head_id - exist_node
                else:
                    head_id = head_id
                x[head_id]
                if 'Al' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 0])
                elif 'Fe' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 1])
                elif 'Co' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 2])
                elif 'Ni' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 3])
                elif 'Cr' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 4])
                elif 'Cu' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 5])
                elif 'Mn' in rel_df.loc[i, ':END_ID']:
                    edge_weight[i] = self.element2weight(x[head_id, 6])

        data_graph_ = Data(x=x, x_id=x_id, x_attr=x_attr, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, y=y, y_mask=y_mask, x_type=x_type)

        if data_graph is None:
            data_graph_, max_hea, min_hea, max_y, min_y = self._normalization_graph(data_graph_, HEA_node_index)
            data_graph = data_graph_
        else:
            data_graph = self._denormalization_graph(data_graph, HEA_node_index_ori, max_hea, min_hea, max_y, min_y)
            data_graph = Data(x=torch.cat((data_graph.x, data_graph_.x), dim=0), 
                            x_id=torch.cat((data_graph.x_id, data_graph_.x_id), dim=0),
                            x_attr=torch.cat((data_graph.x_attr, data_graph_.x_attr), dim=0),
                            edge_index=torch.cat((data_graph.edge_index, data_graph_.edge_index), dim=1), 
                            edge_attr=torch.cat((data_graph.edge_attr, data_graph_.edge_attr), dim=0), 
                            edge_weight=torch.cat((data_graph.edge_weight, data_graph_.edge_weight), dim=0), 
                            y=torch.cat((data_graph.y, data_graph_.y), dim=0), 
                            y_mask=torch.cat((data_graph.y_mask, data_graph_.y_mask), dim=0), 
                            x_type=torch.cat((data_graph.x_type, data_graph_.x_type), dim=0)
                            )
            
            HEA_node_index = HEA_node_index_ori + [_ + exist_node for _ in HEA_node_index]
            data_graph, max_hea, min_hea, max_y, min_y = self._normalization_graph(data_graph, HEA_node_index)

        if active_or_passive is not None:
            if active_or_passive == 0:
                data_graph = self._convert_passive_active(data_graph)

        return node_dict, edge_dict, node_name2id, edge_name2id, data_graph, max_hea, min_hea, max_y, min_y, HEA_node_index
    
    def kFold(self, label, n_splits=5, random_state=None):
        """
        K-Folds cross-validator
        """
        if random_state is not None:
            np.random.seed(random_state)

        graph = self.data_graphs[label].clone()

        if self.active_or_passive[label] == 0:
            graph = self._convert_passive_active(graph)

        k_fold = torch.zeros(graph.y.shape[0], n_splits)
        HEA_node_index = []

        for i in range(graph.y.shape[0]):
            if self.node_dict[label][i]['label'] != 'HEA':
                k_fold[i, :] = -1
            else:
                HEA_node_index.append(i)

        _process_dict = {}
        _structure_dict = {}
        for i in range(graph.edge_index.shape[1]):
            if self.node_dict[label][graph.edge_index[1, i].item()]['label'] == 'process_technology' and self.node_dict[label][graph.edge_index[0, i].item()]['label'] == 'HEA':
                if graph.edge_index[1, i].item() not in _process_dict:
                    _process_dict[graph.edge_index[1, i].item()] = [graph.edge_index[0, i].item()]
                else:
                    _process_dict[graph.edge_index[1, i].item()].append(graph.edge_index[0, i].item())
            
            elif self.node_dict[label][graph.edge_index[1, i].item()]['label'] == 'structure' and self.node_dict[label][graph.edge_index[0, i].item()]['label'] == 'HEA':
                if graph.edge_index[1, i].item() not in _structure_dict:
                    _structure_dict[graph.edge_index[1, i].item()] = [graph.edge_index[0, i].item()]
                else:
                    _structure_dict[graph.edge_index[1, i].item()].append(graph.edge_index[0, i].item())

        _node_part_list = [v for v in _process_dict.values()]
        _node_part_list.extend([v for v in _structure_dict.values()])
        _node_part_list.sort(key=lambda x: len(x))

        if len(_node_part_list) > 0:
            while len(_node_part_list[0]) <= 1:
                _node_part = _node_part_list.pop(0)
                for _node in _node_part:
                    k_fold[_node, :] = -2
                if len(_node_part_list) == 0:
                    break

        if len(_node_part_list) > 0:
            while len(_node_part_list[0]) <= max(8, n_splits):
                order = np.arange(n_splits)
                np.random.shuffle(order)
                _node_part = _node_part_list.pop(0)
                _node_part = np.array(_node_part)
                for i in range(len(_node_part)):
                    if k_fold[_node_part[i], :].sum() <= 0:
                        k_fold[_node_part[i], order[i%len(order)]] = 1
                if k_fold[_node_part, :].sum(axis=0).max() == len(_node_part):
                    for _node in _node_part:
                        k_fold[_node, :] = -2
                if len(_node_part_list) == 0:
                    break
            
        if len(_node_part_list) > 0:
            _node = []
            for _node_part in _node_part_list:
                _node.extend(_node_part)
            _node = list(set(_node))
            _node = [v for v in _node if k_fold[v, :].sum() == 0]
            _k_fold_sum = [k_fold[k_fold.sum(axis=1) > 0, :][graph.y_mask[k_fold.sum(axis=1) > 0, :][:, i] == 1].sum(axis=0) for i in range(graph.y_mask.shape[1])]
            np.random.shuffle(_node)
            for i in range(len(_node)):
                _node_label = graph.y_mask[_node[i], :]
                _node_label = np.array(_node_label)
                _node_label = _node_label.reshape(-1)
                _node_label = np.where(_node_label == 1)[0][0]
                k_fold[_node[i], _k_fold_sum[_node_label].argmin()] = 1
                _k_fold_sum[_node_label][_k_fold_sum[_node_label].argmin()] += 1
        
        if self.active_or_passive[label] == 0:
            graph = self._convert_passive_active(graph)

        graph.k_fold = k_fold
        self.data_graphs[label] = graph

        return self.data_graphs[label]
    
    def spilt_data(self, label, k_fold=True, val_need=True):
        """
        split data set
        """
        graphs = []
        if k_fold:
            k_num = self.data_graphs[label].k_fold.shape[1]
            
            for k in range(k_num):
                train_graph = self.data_graphs[label].clone()
                if val_need and k_num > 2:
                    valid_graph = self.data_graphs[label].clone()
                test_graph = self.data_graphs[label].clone()

                if val_need and k_num > 2:
                    valid_sub_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
                    valid_sub_mask[valid_graph.k_fold[:, k] == 1] = 0

                test_sub_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
                test_sub_mask[test_graph.k_fold[:, k-1] == 1] = 0

                train_sub_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
                if val_need and k_num > 2:
                    train_sub_mask[valid_graph.k_fold[:, k-1] == 1] = 0
                train_sub_mask[test_graph.k_fold[:, k] == 1] = 0

                train_mask = torch.zeros(train_graph.x.shape[0], dtype=torch.bool)
                for j in range(k_num):
                    if (j != k and j != k + k_num) and (j != k-1 and j != k-1 + k_num):
                        train_mask[train_graph.k_fold[:, j] == 1] = 1

                if val_need and k_num > 2:
                    valid_mask = torch.zeros(train_graph.x.shape[0], dtype=torch.bool)
                    valid_mask[valid_graph.k_fold[:, k-1] == 1] = 1
                test_mask = torch.zeros(train_graph.x.shape[0], dtype=torch.bool)
                test_mask[test_graph.k_fold[:, k] == 1] = 1

                train_graph.train_mask = train_mask.clone()
                valid_graph.train_mask = train_mask.clone()
                test_graph.train_mask = train_mask.clone()
                if val_need and k_num > 2:
                    train_graph.valid_mask = valid_mask.clone()
                    valid_graph.valid_mask = valid_mask.clone()
                    test_graph.valid_mask = valid_mask.clone()
                train_graph.test_mask = test_mask.clone()
                valid_graph.test_mask = test_mask.clone()
                test_graph.test_mask = test_mask.clone()

                train_graph = self._subgraph(train_graph, train_sub_mask)
                if val_need and k_num > 2:
                    valid_graph = self._subgraph(valid_graph, valid_sub_mask)
                test_graph = self._subgraph(test_graph, test_sub_mask)

                if val_need and k_num > 2:
                    graphs.append([train_graph, valid_graph, test_graph])
                else:
                    graphs.append([train_graph, test_graph])

            return graphs
        else:
            k_num = self.data_graphs[label].k_fold.shape[1]

            train_graph = self.data_graphs[label].clone()
            if val_need and k_num > 2:
                valid_graph = self.data_graphs[label].clone()
            test_graph = self.data_graphs[label].clone()

            if val_need and k_num > 2:
                valid_sub_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
                valid_sub_mask[valid_graph.k_fold[:, -1] == 1] = 0

            test_sub_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
            test_sub_mask[test_graph.k_fold[:, -2] == 1] = 0

            train_sub_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
            if val_need and k_num > 2:
                train_sub_mask[valid_graph.k_fold[:, -2] == 1] = 0
            train_sub_mask[test_graph.k_fold[:, -1] == 1] = 0

            train_graph = self._subgraph(train_graph, train_sub_mask)
            if val_need and k_num > 2:
                valid_graph = self._subgraph(valid_graph, valid_sub_mask)
            test_graph = self._subgraph(test_graph, test_sub_mask)

            if val_need and k_num > 2:
                return train_graph, valid_graph, test_graph
            else:
                return train_graph, test_graph

    def _subgraph(self, graph, sub_mask):
        """
        Generate the subgraph according to the sub_mask
        """
        sub_mask = sub_mask.reshape(-1)
        subgraph = graph.clone()
        subgraph.x = graph.x[sub_mask == 1, :]
        subgraph.x_id = graph.x_id[sub_mask == 1, :]
        subgraph.x_attr = graph.x_attr[sub_mask == 1, :]
        subgraph.y = graph.y[sub_mask == 1, :]
        subgraph.y_mask = graph.y_mask[sub_mask == 1, :]
        subgraph.x_type = graph.x_type[sub_mask == 1, :]
        if graph.k_fold is not None:
            subgraph.k_fold = graph.k_fold[sub_mask == 1, :]
        if graph.train_mask is not None:
            subgraph.train_mask = graph.train_mask[sub_mask == 1]
        if graph.valid_mask is not None:
            subgraph.valid_mask = graph.valid_mask[sub_mask == 1]
        if graph.test_mask is not None:
            subgraph.test_mask = graph.test_mask[sub_mask == 1]

        subgraph.edge_attr = subgraph.edge_attr[(sub_mask[subgraph.edge_index[0, :]] == 1) & (sub_mask[subgraph.edge_index[1, :]] == 1), :]
        subgraph.edge_weight = subgraph.edge_weight[(sub_mask[subgraph.edge_index[0, :]] == 1) & (sub_mask[subgraph.edge_index[1, :]] == 1)]
        subgraph.edge_index = subgraph.edge_index[:, (sub_mask[subgraph.edge_index[0, :]] == 1) & (sub_mask[subgraph.edge_index[1, :]] == 1)]
        subgraph.edge_index = self._reindex_edge(subgraph.edge_index, sub_mask)

        return subgraph

    def _reindex_edge(self, edge_index, sub_mask):
        """
        Reindex the edge index
        """
        sub_mask = sub_mask.reshape(-1)
        edge_index = edge_index.clone()
        for i in range(edge_index.shape[1]):
            edge_index[0, i] = torch.nonzero(sub_mask[0:edge_index[0, i]] == 1).shape[0]
            edge_index[1, i] = torch.nonzero(sub_mask[0:edge_index[1, i]] == 1).shape[0]
        return edge_index

    def _normalization_graph(self, graph, HEA_node_index):
        """
        Normalization graph
        """
        max_hea = torch.max(torch.max(graph.x[HEA_node_index, :], dim=0)[0], dim=0)[0]
        min_hea = torch.tensor(0.0)
        graph.x[HEA_node_index, :] = (graph.x[HEA_node_index, :] - min_hea) / (max_hea - min_hea)

        max_y = torch.max(graph.y[HEA_node_index, :], dim=0)[0]
        y_ = graph.y.clone()
        y_[graph.y_mask == 0] = 999999
        min_y = torch.min(y_, dim=0)[0]
        graph.y[HEA_node_index, :] = (graph.y[HEA_node_index, :] - min_y) / (max_y - min_y)
        return graph, max_hea, min_hea, max_y, min_y

    def _denormalization_graph(self, graph, HEA_node_index, max_hea, min_hea, max_y, min_y):
        """
        Denormalization graph
        """
        graph.x[HEA_node_index, :] = graph.x[HEA_node_index, :] * (max_hea - min_hea) + min_hea

        graph.y[HEA_node_index, :] = graph.y[HEA_node_index, :] * (max_y - min_y) + min_y
        graph.y[graph.y_mask == 0] = 0

        return graph
    
    def _convert_passive_active(self, graph):
        """
        Convert the direction of the relation
        """
        graph.edge_index = torch.stack((graph.edge_index[1, :], graph.edge_index[0, :]), dim=0)
        return graph
    


if __name__=='__main__':
    pass