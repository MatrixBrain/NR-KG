import os
from GCN_model import GCN
import torch
import torch.nn as nn
from tools.early_stopping import EarlyStopping


def train(model, graph, optimizer, criterion, scheduler, a_loss, device):
    model.train()
    optimizer.zero_grad()
    out, score = model(graph, batch=None)

    y_dim = graph.y.shape[1]
    loss_regression = 0
    if y_dim == 1:
        loss_regression = criterion[0](out[graph.train_mask], graph.y[graph.train_mask])
    else:
        for i in range(y_dim):
            mask = graph.train_mask * graph.y_mask[:, i]
            mask = mask.bool()
            loss_regression += criterion[0](out[:, i][mask], graph.y[:, i][mask])
        loss_regression /= y_dim

    loss_CLL = criterion[1](score[0], score[1], (torch.ones(len(score[0])) * (-1)).to(device))
    loss_PPL = score[2].mean()
    loss = a_loss[0] * loss_regression + a_loss[1] * loss_CLL + a_loss[2] * loss_PPL

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss

def valid(model, graph, criterion, a_loss, device):
    model.eval()
    out, score = model(graph, batch=None)

    y_dim = graph.y.shape[1]
    loss_regression = 0
    if y_dim == 1:
        loss_regression = criterion[0](out[graph.valid_mask], graph.y[graph.valid_mask])
    else:
        for i in range(y_dim):
            mask = graph.valid_mask * graph.y_mask[:, i]
            mask = mask.bool()
            loss_regression += criterion[0](out[:, i][mask], graph.y[:, i][mask])
        loss_regression /= y_dim

    loss_CLL = criterion[1](score[0], score[1], (torch.ones(len(score[0])) * (-1)).to(device))
    loss_PPL = score[2].mean()

    loss = a_loss[0] * loss_regression + a_loss[1] * loss_CLL + a_loss[2] * loss_PPL
    return loss


def train_model_base(data_graph, device, args, fold_k, data_graph_test=None, data_graph_valid=None, state=None, model=None):
    data_graph.to(device)
    data_graph_test.to(device)
    data_graph_valid.to(device)
    embeddings_num = data_graph.x.shape[0] -  (data_graph.x_attr == 1).sum().item()
    edgetype_num = len(torch.unique(data_graph.edge_attr, dim=0))
    out_channels = data_graph.y.shape[1]
    embeddings_dict = {} 
    id = data_graph.x_id[data_graph.x_attr != 1]
    for i in range(len(id)):
        embeddings_dict[id[i].item()] = torch.tensor(i).long().to(device)

    if state == 'train':
        model = GCN(in_channels=data_graph.num_features, 
                    embeddings_num = embeddings_num,
                    edgetype_num = edgetype_num,
                    embeddings_dict=embeddings_dict,
                    hidden_channels=args.hidden_size, out_channels=out_channels, 
                    num_layers=args.num_layers, dropout=args.dropout,
                    device=device, activation=args.activation).to(device)
    else:
        model = model.to(device)
    
    if state == 'train':
        # criterion
        criterion_MSE = nn.MSELoss(reduction='mean').to(device)
        criterion_CLL = nn.MarginRankingLoss(margin=1, reduction='mean').to(device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # scheduler
        if args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        # early stopping
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, delta=0)

        # train
        for epoch in range(1, args.epoch + 1):
            a_CLL_loss_ = args.a_CLL_loss
            a_MSELoss_ = args.a_MSELoss
            a_PPL_Loss_ = args.a_PPL_Loss

            loss = train(model, graph=data_graph, optimizer=optimizer, criterion=[criterion_MSE, criterion_CLL], scheduler=scheduler, a_loss=[a_MSELoss_, a_CLL_loss_, a_PPL_Loss_], device=device)

            if data_graph_valid is not None:
                loss_v = valid(model, graph=data_graph_valid, criterion=[criterion_MSE, criterion_CLL], a_loss=[a_MSELoss_, a_CLL_loss_, a_PPL_Loss_], device=device)
            
            # Validation
            if epoch % 10 == 0:
                # 打印更优雅
                print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, loss, loss_v))
                if not os.path.exists(args.root_path + '/model/' + str(args.id)):
                    os.makedirs(args.root_path + '/model/' + str(args.id))
                torch.save(model, args.root_path + '/model/' + str(args.id) + '/model_' + str(fold_k) + '_lastmodel.pkl')

            # Early stopping
            if args.early_stop:
                early_stopping(loss_v)
                if early_stopping.early_stop:
                    print("Early stopping at epoch: ", epoch)
                    break
    

    model.eval()
    out, score = model(data_graph, batch=None)
    out_test, score_test = model(data_graph_test, batch=None)

    return out, score, out_test, score_test



if __name__ == "__main__":
    pass