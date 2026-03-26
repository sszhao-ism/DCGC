import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import *
from layers import *
from filter import *
from sklearn.decomposition import PCA
import torch
import scipy.sparse as sp
from graph_norm import *
from corss_att import *
from time import *
from scipy import io
#cora
# Namespace(alpha=0.55, beta=1e-20, cluster_num=7, cuda=True, dataset='cora', device='cuda:0', dims=500, dropout=0.0, epochs=400, exp_lr=2e-05, exp_wd=2e-05, gamma=1e-20,
#  gnnlayers=3, gpu=0, hid_dim=2708, hidden=128, hop_num=2, knn=30, lr=0.0003, n_layers=2, no_cuda=False, seed=42, sigma=0.5, threshold=0.95, type=None, v_input=1)

# parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")#4
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128, help='hidden_num')#128
parser.add_argument('--dims', type=int, default=500, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate.') #1e-4
parser.add_argument('--hop_num', type=float, default=2, help='Number of hops')
parser.add_argument('--alpha', type=float, default=0.55, help='Balance parameter for loss function') #0.5
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold for high confidence samples') #0.95
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')#7
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--n_layers', type=int, default='2', help='tiquqi cengshu')
parser.add_argument('--exp_lr', type=float, default=2e-5, help='Learning rate of ICML.')  ## 1e-5
parser.add_argument('--exp_wd', type=float, default=2e-5, help='Weight decay of ICML.') ## 1e-5
parser.add_argument('--type', type=str, help='Type argument')
parser.add_argument('--v_input', type=int, default=1, help='Degree of freedom of T distribution')  ## 1
parser.add_argument('--sigma', type=float, default=0.5, help='Weight parameters for knn.')  ##0.5
parser.add_argument('--knn', type=int, default=30, help='The K of KNN graph.')  ## 155 50-200
parser.add_argument("--hid_dim", type=int, default=2708, help='Hidden layer dim.')  
parser.add_argument('--dropout', type=float, default=0., help='drop rate.')  ## 0.
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--beta', type=float, default=1e-20, help='Weight parameters for loss.')
parser.add_argument('--gamma', type=float, default=1e-20, help='Weight parameters for ICML.')


def set_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Using {} dataset".format(args.dataset))
file = open("result_baseline.csv", "a+")
print(args.dataset, file=file)
file.close() #

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
set_seed(args.seed)
print(args)
begin_time = time()

# Load data
if args.dataset in [ 'bat', 'eat', 'uat','acm','citeseer','cora']:#'amap', 'bat', 'eat', 'uat','corafull'
    feat, label, A = load_data(args.dataset)
    labels = torch.from_numpy(label)
else:
    data = io.loadmat('./data/{}.mat'.format(args.dataset))
    if args.dataset == 'wiki':
        feat = data['fea'].todense()
        A = data['W'].todense()
    elif args.dataset == 'pubmed':
        feat = data['fea']
        A = data['W'].todense()
    else:
        feat = data['fea']
        A = np.mat(data['W'])
    gnd = data['gnd'].T - 1
    labels = torch.from_numpy(gnd[0, :])
    
adj = A
true_labels = labels
features = feat 

features = torch.from_numpy(features).type(torch.FloatTensor)
in_dim = features.shape[1] #
args.N = N = features.shape[0] #399   
features = features.to(args.device)
print(features.shape)  
if features.shape[1] >= args.dims:
    pca = PCA(n_components=int(args.dims))
else:
    pca = PCA(n_components=features.shape[1]-1)
features = features.cpu().numpy()  
features = pca.fit_transform(features)

features = torch.FloatTensor(features) 
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape) 
if not sp.issparse(adj):
    adj = sp.csr_matrix(adj)
adj.eliminate_zeros()
adj_dense = adj.todense()  # Convert to dense matrix
adj_tensor = torch.tensor(adj_dense, dtype=torch.float32) 
adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
sm_fea_s = sp.csr_matrix(features).toarray() 
for a in adj_norm_s:  # type: ignore  
    sm_fea_s = a.dot(sm_fea_s)  
sm_fea_s = torch.FloatTensor(sm_fea_s)   
MLP_model = MLP_model([features.shape[1]])  # MLP model
MLP_model = MLP_model.cuda()
optimizer_mlp = optim.Adam(MLP_model.parameters(), lr=args.lr)  # Model's optimizer
args.N = N =features.shape[0] #2708
norm_factor, edge_index, edge_weight, adj_norm, knn, Lap = cal_norm(adj, args, features)
Lap_Neg = cal_Neg(adj_norm, knn, args)
features_att = features.to(args.device)

AGCN_Net = AGCN_Net(N, args).cuda()
optimizer_att = optim.Adam([{'params':AGCN_Net.params_exp,'weight_decay':args.exp_wd, 'lr': args.exp_lr}])

acc_list = []
nmi_list = []
ari_list = []
f1_list = []
EYE = torch.eye(args.N).to(args.device)

for seed in range(10):
    setup_seed(seed)
    best_acc, best_nmi, best_ari, best_f1, predict_labels, centers = clustering(sm_fea_s, true_labels, args.cluster_num)
    model = Encoder_net([features.shape[1]] + [args.dims]) #500+500
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()
        inx = sm_fea_s.cuda() 

    best_acc = 0
    ident = torch.eye(sm_fea_s.shape[0]).cuda()
    target = torch.eye(sm_fea_s.shape[0]).cuda()
    feat = torch.tensor(features).cuda().clone().detach().requires_grad_(True)


    for epoch in tqdm(range(args.epochs)):
        model.train()
        AGCN_Net.train()
        optimizer_att.zero_grad()
        x_learn = feat 
        inx_2 = x_learn 
        atten_adj = AGCN_Net(knn,adj_norm) 
        H = ident - atten_adj
        for i in range(args.gnnlayers):
            inx_2 = H @ inx_2
        a_loss = -F.mse_loss(atten_adj, adj_tensor.cuda())
        aug_loss = a_loss # 
        F1 = model(inx) #
        F2 = model(inx_2)  
        A_loop = add_self_loops(adj_tensor.cuda())
        S = F1 @ F2.T
        S_loss = F.mse_loss(A_loop, S.to(args.device))
        
        loss_L =( (args.gamma * torch.trace(torch.mm(torch.mm(inx_2.t(), Lap), inx_2)) )\
                        - args.beta*(torch.trace(torch.mm(torch.mm(inx_2.t(), Lap_Neg), inx_2))))
        infoNCE = loss_cal(F1, F2)
        loss = infoNCE + args.alpha * aug_loss + S_loss  +loss_L
        loss.backward()
        optimizer.step()
        optimizer_att.step()
        optimizer_mlp.step()
        # Epoch > 200, second stage --> Refine
        if epoch > 200:
            if epoch % 20 == 0:
                # select high-confidence samples
                distribute = F.softmax(torch.sum(torch.pow(((F1 + F2) / 2).unsqueeze(1) - centers, 2), 2), dim=1)
                distribute = torch.min(distribute, dim=1).values
                value, index = torch.topk(distribute, int(len(distribute) * (args.threshold)))
                distribute = torch.where(distribute <= value[-1], torch.ones_like(distribute),
                                         torch.zeros_like(distribute))
                pseudo_label_index = torch.nonzero(distribute).reshape(-1, )
                matrix_index = np.ix_(pseudo_label_index.cpu(), pseudo_label_index.cpu())
                predict_labels = torch.tensor(predict_labels).cuda()
                pseudo_matrix = (predict_labels == predict_labels.unsqueeze(1)).float().cuda()
                S = F1 @ F2.T
                S = normalize(S)

                # refine
                atten_adj[matrix_index] = atten_adj[matrix_index] * pseudo_matrix[matrix_index]
                atten_adj = atten_adj * S.detach()

                inx_2 = x_learn
                H_2 = ident - atten_adj
                for a in range(args.gnnlayers):
                    inx_2 = H @ inx_2
                inx_new_2 = inx_2
                inx_new_2 = inx_new_2.float().cuda()

                if epoch % 10 == 0:
                    model.eval()
                    F_1 = model(inx)
                    F_new_2 = model(inx_new_2)
                    hidden_emb = (F_1 + F_new_2) / 2
                    acc, nmi, ari, f1, predict_labels, centers = clustering(hidden_emb, true_labels, args.cluster_num)
                    
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
        else:
            if epoch % 2 == 0:
                model.eval()
                F1 = model(inx)
                F2 = model(inx_2)
                hidden_emb = (F1 + F2) / 2
                acc, nmi, ari, f1, predict_labels, centers = clustering(hidden_emb, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}, best_f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
    file = open("result_baseline.csv", "a+")
    print(best_acc, best_nmi, best_ari, best_f1, file=file)
    file.close()

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
file = open("result_baseline.csv", "a+")
print(args.gnnlayers, args.lr, args.dims, args.sigma, file=file)
print(acc_list.mean(), acc_list.std())
print(nmi_list.mean(), nmi_list.std())
print(ari_list.mean(), ari_list.std())
print(f1_list.mean(), f1_list.std())
file.close()
