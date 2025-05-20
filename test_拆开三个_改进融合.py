import time
from itertools import cycle
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataload_excelremove import DataSet, load_data_remove
import gc
import pandas as pd
import matplotlib.pyplot as plt

from config import get_args
#from cnn3dend_21_smt_parallel3_CHAI import feature_Net4s_tp4_feature
#from cnn3dend_21_smt_parallel3_CHAI import feature_Net4s_tp4
from 融合改进4 import feature_Net4s_tp4_feature
from 融合改进4 import feature_Net4s_tp4
from 融合改进4 import feature_Net4s_tp4_ferturex

def test_model(test_data, ResNet_3D_label, ResNet_3D_fenshu,ResNet_3D_tezheng):
    ResNet_3D_label.eval()
    ResNet_3D_fenshu.eval()
    ResNet_3D_tezheng.eval()
    n_batches1 = len(test_data)
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    #ResNet_3D_label.load_state_dict(torch.load("./model/3Dfeature_Net4s_tp4_feature_epoch_68_d24.pt"))
    #ResNet_3D_fenshu.load_state_dict(torch.load("./model/3Dfeature_Net4s_tp4_epoch_68_d24.pt"))
    #ResNet_3D_tezheng.load_state_dict(torch.load("./model/3Dfeature_Net4s_tp4_ferturex_epoch_68_d24.pt"))
    with torch.no_grad():
	    for (images, labels) in tqdm(test_data, total=n_batches1):
		    images = images.cuda()
		    # labels = labels.cuda()
		    labels = labels.cuda()
		    images = images.to(torch.float32)
		    output1, output2, output3 = ResNet_3D_fenshu(images)
		    feature_x0 = ResNet_3D_tezheng(images)
		    output, out_featurex,crw= ResNet_3D_label(output1, output2, output3,labels, images,feature_x0)
		    _, predicted = torch.max(output, 1)
		    true_label.extend(list(labels.cpu().flatten().numpy()))
		    data_pre.extend(list(predicted.cpu().flatten().numpy()))
		    total += labels.size(0)
		    correct += (predicted == labels).sum().item()

    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('The result of test data: \n')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    del correct, total, true_label, data_pre, images, labels, output, TN, FP, FN, TP
    gc.collect()
    # return ACC, SEN, SPE, AUC


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    root_path_train = 'E:/shuju_at/Mribr21_fen_caijian/train/'
    path1 = 'AD/'
    path2 = 'CN/'
    path3 = 'PMCI/'
    path4 = 'SMCI/'
    root_path_val = 'E:/shuju_at/Mribr24_fen_caijian/val/'
    root_path_test = 'E:/shuju_at/Mribr24_fen_caijian/test/'
    test_data = load_data_remove(args, root_path_test, path1, path2)

    ResNet_3D_label = feature_Net4s_tp4_feature().cuda()
    ResNet_3D_fenshu=feature_Net4s_tp4().cuda()
    ResNet_3D_tezheng=feature_Net4s_tp4_ferturex().cuda()
    # 将模型用于测试集

    ResNet_3D_label.load_state_dict(torch.load("./model/3Dfeature_Net4s_tp4_feature_epoch_68_d24.pt"))
    ResNet_3D_fenshu.load_state_dict(torch.load("./model/3Dfeature_Net4s_tp4_epoch_68_d24.pt"))
    ResNet_3D_tezheng.load_state_dict(torch.load("./model/3Dfeature_Net4s_tp4_ferturex_epoch_68_d24.pt"))
    test_model(test_data, ResNet_3D_label,ResNet_3D_fenshu,ResNet_3D_tezheng)

