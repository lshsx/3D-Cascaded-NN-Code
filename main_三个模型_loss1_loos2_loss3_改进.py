import time
from itertools import cycle
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataloader_1 import DataSet, load_data
from dataload_excelremove import load_data_remove
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
# from resnet_3dsepo4se5 import get_net
from config import get_args
#from resnet_3dpo4 import get_net
# from cnn3dend_21_smt import feature_Net4s_tp4
from 融合改进4 import feature_Net4s_tp4_feature
from 融合改进4 import feature_Net4s_tp4_ferturex
from 融合改进4 import feature_Net4s_tp4

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from tqdm import tqdm

def train_epoch(epoch, ResNet_3D_label, ResNet_3D_fenshu, ResNet_3D_tezheng, train_data, train_log_file, weight_decay=None, accumulate_loss=False):
    ResNet_3D_label.train()
    ResNet_3D_fenshu.train()
    ResNet_3D_tezheng.train()
    
    n_batches = len(train_data)
    print(n_batches)

    # Learning rate schedule
    if epoch < 40:
        LEARNING_RATE = 0.0001
    else:
        LEARNING_RATE = 0.00001

    # Optimizer setup with optional weight decay
    optimizer_params = [
        {'params': ResNet_3D_label.parameters(), 'lr': LEARNING_RATE, "weight_decay": weight_decay if weight_decay else 0},
        {'params': ResNet_3D_fenshu.parameters(), 'lr': LEARNING_RATE, "weight_decay": weight_decay if weight_decay else 0},
        {'params': ResNet_3D_tezheng.parameters(), 'lr': LEARNING_RATE, "weight_decay": weight_decay if weight_decay else 0}
    ]
    optimizer = torch.optim.Adam(optimizer_params, lr=LEARNING_RATE)

    # Loss functions
    criterion_reg_adas = nn.MSELoss()
    criterion_reg_cdrsb = nn.MSELoss()
    criterion_reg_mmse = nn.MSELoss()

    total_loss_label = 0
    total_loss_fenshu = 0
    total_cross = 0
    source_correct = 0
    total_label = 0

    for (images, adas11, cdrsb, mmse, label) in tqdm(train_data, total=n_batches):
        images = images.cuda()
        adas11 = adas11.cuda()
        cdrsb = cdrsb.cuda()
        mmse = mmse.cuda()
        label = label.cuda()
        
        # Convert data to float
        images, adas11, cdrsb, mmse = images.float(), adas11.float(), cdrsb.float(), mmse.float()

        # Model forward passes
        output1, output2, output3 = ResNet_3D_fenshu(images)
        faature_x0 = ResNet_3D_tezheng(images)
        output, out_featurex, cross00 = ResNet_3D_label(output1, output2, output3, label, images, faature_x0)

        # Compute losses
        loss_adas = criterion_reg_adas(output1[:, 0], adas11)
        loss_cdrsb = criterion_reg_cdrsb(output2[:, 0], cdrsb)
        loss_mmse = criterion_reg_mmse(output3[:, 0], mmse)
        loss_label = F.cross_entropy(output, label)

        loss_predit_fenshu = 0.001 * (loss_adas + loss_cdrsb + loss_mmse)
        constrative_coefficient = get_args().contrastive_coefficient
        loss_cross = constrative_coefficient * cross00

        _, preds = torch.max(output, 1)
        source_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
        total_label += label.size(0)

        # Reset gradients
        optimizer.zero_grad()

        # **Accumulate losses if accumulate_loss=True**
        if accumulate_loss:
            total_loss = loss_label + loss_predit_fenshu + loss_cross  # Sum all losses
            total_loss.backward()  # Single backward pass
        else:
            loss_label.backward(retain_graph=True)
            loss_predit_fenshu.backward(retain_graph=True)
            loss_cross.backward()

        optimizer.step()

        # Track losses for logging
        total_loss_label += loss_label.item()
        total_loss_fenshu += loss_predit_fenshu.item()
        total_cross += cross00.item()

    # Compute final metrics
    acc = source_correct / total_label
    mean_loss_label = total_loss_label / n_batches
    mean_loss_fenshu = total_loss_fenshu / n_batches
    mean_loss_cross = total_cross / n_batches

    # Print training results
    print(f'Epoch: [{epoch:2d}], '
          f'Label Loss: {mean_loss_label:.6f}, '
          f'Fenshu Loss: {mean_loss_fenshu:.6f}, '
          f'Train Accuracy: {acc:.4f}, '
          f'Cross Loss: {mean_loss_cross:.6f}')

    # Log training results
    log_str = (
        f'Epoch: {epoch} '
        f'Label Loss: {mean_loss_label:.6f} '
        f'Fenshu Loss: {mean_loss_fenshu:.6f} '
        f'Train Accuracy: {acc:.4f} '
        f'Cross Loss: {mean_loss_cross:.6f}\n'
    )
    train_log_file.write(log_str)

    # Clean up
    del acc, train_data, n_batches
    gc.collect()

    return mean_loss_label, mean_loss_fenshu, source_correct, total_label

def val_model(epoch, val_data, ResNet_3D_label,ResNet_3D_fenshu, ResNet_3D_tezheng,val_log_file):
    ResNet_3D_label.eval()
    ResNet_3D_fenshu.eval()
    ResNet_3D_tezheng.eval()
    n_batches1=len(val_data)
    print('Validate a model on the val data...')
    correct = 0
    total = 0
    total_loss_label = 0
    total_loss_fenshu = 0
    total_cross = 0
    true_label = []
    data_pre = []
    criterion_reg_adas = nn.MSELoss()
    criterion_reg_adas = criterion_reg_adas.to(torch.float32)
    criterion_reg_cdrsb = nn.MSELoss()
    criterion_reg_cdrsb = criterion_reg_cdrsb.to(torch.float32)
    criterion_reg_mmse = nn.MSELoss()
    criterion_reg_mmse = criterion_reg_mmse.to(torch.float32)
    with torch.no_grad():
        for (images, adas11, cdrsb, mmse, labels) in tqdm(val_data, total=n_batches1):
            images = images.cuda()
            # labels = labels.cuda()
            adas11 = adas11.cuda()
            cdrsb = cdrsb.cuda()
            mmse = mmse.cuda()
            labels = labels.cuda()
            images, adas11, cdrsb, mmse = images.to(torch.float32), adas11.to(torch.float32), cdrsb.to(torch.float32), mmse.to(torch.float32)
            output1, output2, output3 = ResNet_3D_fenshu(images)
            feature_x0= ResNet_3D_tezheng(images)
            output, out_featurex,cross00= ResNet_3D_label(output1, output2, output3,labels, images,feature_x0)
            loss_adas = criterion_reg_adas(output1[:, 0], adas11)
            loss_cdrsb = criterion_reg_cdrsb(output2[:, 0], cdrsb)
            loss_mmse = criterion_reg_mmse(output3[:, 0], mmse)
            loss_label = F.cross_entropy(output, labels)
            loss_fenshu =0.001*(loss_adas+loss_cdrsb+loss_mmse)
            total_loss_label += loss_label.item()
            total_loss_fenshu += loss_fenshu.item()
            total_cross += cross00.item()
            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    mean_loss_label = total_loss_label / len(val_data)
    mean_loss_fenshu= total_loss_fenshu / len(val_data)
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    log_str = 'Epoch: ' + str(epoch) \
              + '\n' \
              + 'TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN: ' + str(FN) \
              + '  ACC:  ' + str(ACC) \
              + '  SEN:  ' + str(SEN) \
              + '  SPE:  ' + str(SPE) \
              + '  AUC:  ' + str(AUC) \
              + '\n'
    val_log_file.write(log_str)
    del correct, total, true_label, data_pre, images, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss_label,mean_loss_fenshu  #, feature_for_GCN


def test_model(epoch, test_data, ResNet_3D_label, ResNet_3D_fenshu,ResNet_3D_tezheng, test_log_file):
    ResNet_3D_label.eval()
    ResNet_3D_fenshu.eval()
    ResNet_3D_tezheng.eval()

    n_batches1 = len(test_data)
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    with torch.no_grad():
        for (images, labels) in tqdm(test_data, total=n_batches1):
            images = images.cuda()
            # labels = labels.cuda()
            labels = labels.cuda()
            images= images.to(torch.float32)
            output1, output2, output3 = ResNet_3D_fenshu(images)
            feature_x0= ResNet_3D_tezheng(images)
            output, out_featurex,cross00= ResNet_3D_label(output1, output2, output3, labels, images,feature_x0)
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
    log_str = f'Epoch: {epoch},TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN},\
    \nACC: {ACC}, SEN: {SEN}, SPE: {SPE}, AUC: {AUC}\n'

    test_log_file.write(log_str)
    del correct, total, true_label, data_pre, images, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC



if __name__ == '__main__':
    DROPOUT = 0.3
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if get_args().val_test_turbo:
        root_path = "MRIbr24_val_test_turbo/"
    else:
        root_path = "MRIbr24/"

    root_path_train = root_path + 'train/'
    root_path_val = root_path + 'val/'
    root_path_test = root_path + 'test/'
    path1 = 'AD/'
    path2 = 'CN/'
    path3 = 'PMCI/'
    path4 = 'SMCI/'
    excel_file = 'ADlist/Allm00ft_score.xlsx'
    print(f"Loading data from {root_path_train}, {root_path_val}, {root_path_test}")
  
    train_data = load_data(args, root_path_train, path1,path2, excel_file, oversampling=get_args().oversampling)

    val_data = load_data(args, root_path_val, path1,path2, excel_file)
    test_data = load_data_remove(args, root_path_test, path1,path2)
    #class_preditcnn = feature_Net4s_tp4().cuda()

    
    ResNet_3D_label = feature_Net4s_tp4_feature(DROPOUT, args.contrastive_loss, args.learnable_alpha).cuda()



    ResNet_3D_fenshu = feature_Net4s_tp4().cuda()
    ResNet_3D_tezheng=feature_Net4s_tp4_ferturex().cuda()

    train_best_loss_label = 10000
    train_best_loss_fenshu=10000
    val_best_loss_label = 10000
    val_best_loss_fenshu=10000
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0
    t_precision = 0
    t_f1 = 0



    count = 0

    since = time.time()




    saving_suffix = ""

    if get_args().contrastive_loss:

        saving_suffix += "_contrastive_loss"

    if get_args().learnable_alpha:
        saving_suffix += "_learnable_alpha"

    if get_args().contrastive_coefficient != 1.0:

        saving_suffix += f"_{get_args().contrastive_coefficient}"
    
    if get_args().oversampling:

        saving_suffix += "_oversampling"
    
    if get_args().add_weight_decay:

        saving_suffix += "_add_weight_decay"
    
    if get_args().val_test_turbo:
        saving_suffix += "_val_test_turbo"
    
    if get_args().accumulate_loss:

        saving_suffix += "_accumulate_loss"



    DIR = "ADD_MORE_TEST1"
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    LOG_SAVING_DIR = os.path.join(DIR, "Log1")
    TRAIN_CSV_DIR = os.path.join(DIR, "Train_csv1")
    FIGURE_SAVING_DIR = os.path.join(DIR, "Figure1")
    MODEL_SAVING_DIR = os.path.join(DIR, "Model1")


    fenshu_model_dir = os.path.join(MODEL_SAVING_DIR, saving_suffix,"fenshu")
    label_model_dir = os.path.join(MODEL_SAVING_DIR, saving_suffix,"label")
    tezheng_model_dir = os.path.join(MODEL_SAVING_DIR, saving_suffix,"tezheng")

    # Create directories if they do not exist
    os.makedirs(LOG_SAVING_DIR, exist_ok=True)
    os.makedirs(TRAIN_CSV_DIR, exist_ok=True)
    os.makedirs(FIGURE_SAVING_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVING_DIR, exist_ok=True)

    os.makedirs(os.path.join(LOG_SAVING_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(LOG_SAVING_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(LOG_SAVING_DIR, "test"), exist_ok=True)

    os.makedirs(os.path.join(MODEL_SAVING_DIR, saving_suffix), exist_ok=True)
    os.makedirs(fenshu_model_dir, exist_ok=True)
    os.makedirs(label_model_dir, exist_ok=True)
    os.makedirs(tezheng_model_dir, exist_ok=True)



    train_log = os.path.join(LOG_SAVING_DIR, "train", "train_log" + saving_suffix)
    val_log = os.path.join(LOG_SAVING_DIR, "val", "val_log" + saving_suffix)
    test_log = os.path.join(LOG_SAVING_DIR, "test","test_log" + saving_suffix)
    figure_name = os.path.join(FIGURE_SAVING_DIR, "Accuracy_Loss" + saving_suffix)
    train_process_csv = os.path.join(TRAIN_CSV_DIR, "train_process" + saving_suffix)

    # if there are four files already, delete them
    if os.path.exists(f"{train_log}.txt"):
        os.remove(f"{train_log}.txt")
    else:
        os.makedirs(os.path.dirname(train_log), exist_ok=True)

    if os.path.exists(f"{val_log}.txt"):
        os.remove(f"{val_log}.txt")
    else:
        os.makedirs(os.path.dirname(val_log), exist_ok=True)

    if os.path.exists(f"{test_log}.txt"):
        os.remove(f"{test_log}.txt")
    else:
        os.makedirs(os.path.dirname(test_log), exist_ok=True)

    if os.path.exists(f"{train_process_csv}.csv"):
        os.remove(f"{train_process_csv}.csv")
    else:
        os.makedirs(os.path.dirname(train_process_csv), exist_ok=True)
    

    train_loss_all_label = []
    train_loss_all_fenshu = []
    train_acc_all = []
    val_loss_all_label = []
    val_loss_all_fenshu=[]
    val_acc_all = []

    test_acc_all = []
    test_sen_all = []

    test_spe_all = []
    test_auc_all = []

    for epoch in range(1, args.nepoch + 1):
        
        train_log_file = open(f"{train_log}.txt", "a")
        val_log_file = open(f'{val_log}.txt', 'a')

        test_log_file = open(f'{test_log}.txt', 'a')
        # 训练模型
        if get_args().add_weight_decay:
            weight_decay = get_args().weight_decay
        else:
            weight_decay = None

        train_loss_label, train_loss_fenshu, train_correct, len_train = train_epoch(epoch, ResNet_3D_label, 
                                                                        ResNet_3D_fenshu,ResNet_3D_tezheng, train_data, 
                                                                        train_log_file, weight_decay, get_args().accumulate_loss)
        

        # 打印当前训练损失、最小损失和准确率
        if train_loss_label < train_best_loss_label:
            train_best_loss_label = train_loss_label

        if train_loss_fenshu < train_best_loss_fenshu:
            train_best_loss_fenshu = train_loss_fenshu

        train_acc = 100. * train_correct / len_train
        if train_acc > train_best_acc:
            train_best_acc = train_acc
        print('current loss label: ', train_loss_label, 'the best loss label: ', train_best_loss_label)
        print('current loss fenshu: ', train_loss_fenshu, 'the best loss fenshu: ', train_best_loss_fenshu)
        print(f'train_correct/train_data: {train_correct}/{len_train} accuracy: {train_acc:.2f}%')

        # 模型用于验证集并将评价指标写入txt
        #ACC, SEN, SPE, AUC, val_loss, feature_for_GCN = val_model(epoch, val_data, ResNet_3D, val_log_file)
        ACC, SEN, SPE, AUC, val_loss_label, val_loss_fenshu = val_model(epoch, val_data, ResNet_3D_label, ResNet_3D_fenshu,ResNet_3D_tezheng, val_log_file)
        if ACC > val_best_acc:  # if val_loss < val_best_loss   特征保存条件
            target_best_acc = ACC  # val_best_loss = val_loss
            val_best_acc = target_best_acc
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC
            #np.savez('feature_' + str(count), feature_for_GCN.detach().cpu().numpy())
            #count = count+1

        val_log_file.write('The best result:\n')
        val_log_file.write('ACC:  ' + str(val_best_acc) + '  SEN:  ' + str(t_SEN) + '  SPE:  ' + str(
            t_SPE) + '  AUC:  ' + str(t_AUC) + '\n\n')

        print(f'The train acc of this epoch: {train_acc:.2f}%')
        print(f'The best acc: {train_best_acc:.2f}% \n')
        train_log_file.write('train_acc: '+str(train_acc)+' The current total loss label: '+str(train_loss_label)+' The best loss: '+str(train_best_loss_label)+'\n\n')


        # test each model after epoch > 40 from which the model is kind of stable
        if epoch >= 35:
            test_acc, test_sen, test_spe, test_auc = test_model(epoch, test_data, ResNet_3D_label, ResNet_3D_fenshu,ResNet_3D_tezheng, test_log_file)
            print(f'Test results: ACC: {test_acc:.2f}% SEN: {test_sen:.2f}% SPE: {test_spe:.2f}% AUC: {test_auc:.2f}%')
                    # 保存模型
            print('model saved...')
            print()
            label_model_path = os.path.join(label_model_dir, f"3Dfeature_Net4s_tp4_label_epoch_{epoch}_d24.pt")
            fenshu_model_path = os.path.join(fenshu_model_dir, f"3Dfeature_Net4s_tp4_epoch_{epoch}_d24.pt")
            tezheng_model_path = os.path.join(tezheng_model_dir, f"3Dfeature_Net4s_tp4_ferturex_epoch_{epoch}_d24.pt")
            torch.save(ResNet_3D_label.state_dict(), label_model_path)
            torch.save(ResNet_3D_fenshu.state_dict(), fenshu_model_path)
            torch.save(ResNet_3D_tezheng.state_dict(), tezheng_model_path)
        else:
            
            test_acc, test_sen, test_spe, test_auc = None, None, None, None
        


        # 保存训练结果用于画图
        train_loss_all_label.append(train_loss_label)
        train_loss_all_fenshu.append(train_loss_fenshu)
        train_acc_all.append(train_acc)
        val_loss_all_label.append(val_loss_label)
        val_loss_all_fenshu.append(val_loss_fenshu)
        val_acc_all.append(ACC)

        test_acc_all.append(test_acc)
        test_sen_all.append(test_sen)
        test_spe_all.append(test_spe)
        test_auc_all.append(test_auc)
        

        del train_loss_label, train_correct, len_train, train_acc
        gc.collect()
        train_log_file.close()
        val_log_file.close()
    

    # 将模型用于测试集
    # test_acc = test_model(epoch, test_data, ResNet_3D_label, ResNet_3D_fenshu,ResNet_3D_tezheng, test_log_file)
    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all_label": train_loss_all_label,
              "train_loss_all_fenshu": train_loss_all_fenshu,
              "train_acc_all": train_acc_all,
              "val_loss_all_label": val_loss_all_label,
              "val_loss_all_fenshu": val_loss_all_fenshu,
              "val_acc_all": val_acc_all,
              "test_acc_all": test_acc_all,
              "test_sen_all": test_sen_all,
              "test_spe_all": test_spe_all,
               "test_auc_all": test_auc_all,
}
    )
    train_process.to_csv(f'{train_process_csv}.csv')
    test_log_file.close()
    # 画图
    plt.figure(figsize=(12,4))
    plt.subplot(2,2,1)
    plt.plot(train_process.epoch, train_process.train_loss_all_label, label="Train loss label")
    plt.plot(train_process.epoch, train_process.val_loss_all_label, label="Val loss label")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(train_process.epoch, train_process.train_loss_all_fenshu, label="Train loss fenshu")
    plt.plot(train_process.epoch, train_process.val_loss_all_fenshu, label="Val loss fenshu")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2,2,3)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, label="Val acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.savefig(f"{figure_name}.png")
    #plt.show()


