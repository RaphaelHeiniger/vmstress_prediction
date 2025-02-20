from pathlib import Path
import torch
import random
import numpy as np
from model_a.model import MeshGraphNet

def ini_model():
    total_samples = 750
    train_samples = int(total_samples * 0.9*0.8)
    test_samples = int(total_samples * 0.9*0.2)

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for args in [
            {'model_type': 'meshgraphnet',
            'num_layers': 32,
            'batch_size': 16,
            'hidden_dim': 32,
            'epochs': 1000,
            'opt': 'adam',
            'opt_scheduler': 'none',
            'opt_restart': 0,
            'weight_decay': 5e-4,
            'lr': 0.0001,
            'train_size': train_samples,
            'test_size': test_samples,
            'device':'cuda',
            'shuffle': True,
            'save_vmstress_val': True,
            'save_best_model': True,
            'checkpoint_dir': './data/best_models/',
            'postprocess_dir': './data/2d_loss_plots/'},
        ]:
            args = objectview(args)

    args.device = torch.device('cpu')
    torch.manual_seed(5)  #Torch
    random.seed(5)        #Python
    np.random.seed(5)     #NumPy

    model_dir = Path("model_a/model_nl32_bs16_hd32_ep1000_wd0.0005_lr0.0001_shuff_True_tr540_te135.pt")
    model_dir = Path("model_a/model_nl32_bs16_hd32_ep10000_wd0.0005_lr0.0001_shuff_True_tr525_te150.pt")
    PATH = model_dir
    num_node_features = 5
    num_edge_features = 3
    num_classes = 1
    model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(args.device)
    return model, args

def get_prediction(loader, best_model, args):
    print(loader)
    best_model.eval()
    device = args.device
    stats_list = [torch.tensor([1.2513e+02, 2.4880e+02, 1.1460e-02, 0.0000e+00, 1.1173e+00]),
                torch.tensor([7.0174e+01, 1.3741e+02, 1.0644e-01, 1.0000e-08, 1.0511e+01]),
                torch.tensor([0.0000, 0.0000, 8.1152]),
                torch.tensor([6.3235, 6.3771, 3.8468]),
                torch.tensor([0.]),
                torch.tensor([1.])]
    
    for data in loader:
        data = data.to(device)
        break
    print("Node features (x) shape:", data.x.shape)
    print("Edge indices shape:", data.edge_index.shape)
    print("Edge attributes shape:", data.edge_attr.shape)
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device), std_vec_x.to(device), 
        mean_vec_edge.to(device), std_vec_edge.to(device), 
        mean_vec_y.to(device), std_vec_y.to(device)
    )

    with torch.no_grad():
        pred = best_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
    print("Prediction shape", pred.shape)
    print("Prediction", pred)
    return pred
