import csv
import os
import numpy as np
import torch
import argparse
import random
from tools.evaluate import evaluate
from tools.loadGraph import *
from tools.train_model_base import train_model_base
import warnings

warnings.filterwarnings('ignore')



def setup_seed(seed):
    """Set up the seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parameters setting
parser = argparse.ArgumentParser()

parser.add_argument('--state', type=str, default=r'train', help='The state of the model: train or test, default is train')
parser.add_argument('--root_path', type=str, default=r'./', help='The path of the project')
parser.add_argument('--net_path', type=str, default=r'./model/pre_trained/HEA-HD', help='The pre-trained model path')
parser.add_argument('--dataaddr', type=str, default=r'./dataset', help='The path of the dataset')
parser.add_argument('--KGtype', type=str, default='HEA-CRD-KG', help='KG-type: HEA-HD-KG or HEA-CRD-KG')
parser.add_argument('--n_fold', type=int, default=6, help='The number of folds, default is 6')
parser.add_argument('--seed', type=int, default=6666, help='Seed')
parser.add_argument('--data_seed', type=int, default=42, help='Seed for data split')
parser.add_argument('--id', type=int, default=10, help='The id of the experiment')
parser.add_argument('--gpu', type=int, default=-1, help='The id of the gpu, -1 means cpu, default is -1')
parser.add_argument('--epoch', type=int, default=2000, help='The number of epochs, default is 2000')
parser.add_argument('--fp16', type=str2bool, default=False, help='Whether to use fp16, default is False')
parser.add_argument('--early_stop', type=str2bool, default=True, help='Whether to use early stop, default is True')
parser.add_argument('--patience', type=int, default=50, help='The patience of early stop, default is 50')

parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size of the model')
parser.add_argument('--num_layers', type=int, default=2, help='num_layers of the model')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout of the model')
parser.add_argument('--activation', type=str, default='ReLU', help='activation of the model: ReLU, LeakyReLU, PReLU, Tanh, ELU, sigmoid') 

parser.add_argument('--lr', type=float, default=0.01, help='learning rate of the training process')
parser.add_argument('--scheduler', type=str, default='MultiStepLR', help='scheduler: MultiStepLR, StepLR, ExponentialLR')
parser.add_argument('--milestones', nargs='+', type=int, default=[500, 2000, 3500], help='milestones of the scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma of the scheduler')
parser.add_argument('--step_size', type=int, default=500, help='step_size of the scheduler')

parser.add_argument('--a_CLL_loss', type=float, default=1.0, help='Proportion of CLL_loss')
parser.add_argument('--a_MSELoss', type=float, default=1.0, help='Proportion of MSELoss')
parser.add_argument('--a_PPL_Loss', type=float, default=0.1, help='Proportion of PPL_Loss')
parser.add_argument('--loss_step', nargs='+', type=float, default=[0, 1, 0.0], help='Proportion type of loss_step')

args = parser.parse_args()

# initialize
setup_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
root_path = args.root_path
torch.set_default_dtype(torch.float16 if args.fp16 else torch.float32)


# Prepare the data
HEAGraph = Graph_data(KGtype=args.KGtype)
path = r'{}\{}'.format(args.dataaddr, args.KGtype)
graph1 = HEAGraph.load_graph_from_neo4jcsv(path, label='HEA', node_type=0)
HEAGraph.convert_passive_active(label='HEA', active_or_passive='passive')

HEAGraph.kFold('HEA', n_splits=6, random_state=args.data_seed)
for i, k in enumerate(HEAGraph.spilt_data('HEA', k_fold=True)):
    print('Fold: ', i)
    train_graph = k[0]
    valid_graph = k[1]
    test_graph = k[2]

    if args.state == 'train':
        model = None
    else:
        model = torch.load(args.net_path + '/model_{}_lastmodel.pkl'.format(i), map_location=device)
        model.eval()

    # Train the model/Test the model
    out, score, out_test, score_test = train_model_base(train_graph, device, args, fold_k=i, data_graph_test=test_graph, data_graph_valid=valid_graph, state=args.state, model=model)

    # Denormalization
    normalization_param = HEAGraph.get_normalization_param(label='HEA')
    y_max = normalization_param['max_y'].cpu().detach().numpy()
    y_min = normalization_param['min_y'].cpu().detach().numpy()

    out = (out.cpu().detach().numpy() * (y_max - y_min) + y_min)[train_graph.train_mask.cpu().detach().numpy()]
    out_test = (out_test.cpu().detach().numpy() * (y_max - y_min) + y_min)[test_graph.test_mask.cpu().detach().numpy()]
    ground_truth = (train_graph.y.cpu().detach().numpy() * (y_max - y_min) + y_min)[train_graph.train_mask.cpu().detach().numpy()]
    ground_truth_test = (test_graph.y.cpu().detach().numpy() * (y_max - y_min) + y_min)[test_graph.test_mask.cpu().detach().numpy()]

    # Evaluate the model on the test set and save the results
    evaluate(ground_truth_test, out_test, root_path + '/res/' + str(args.id) + '.csv', i, args)

