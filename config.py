import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--AD_dir', type=str,
                        help='subfloder of train or test dataset', default='AD/')
    parser.add_argument('--CN_dir', type=str,
                        help='subfloder of train or test dataset', default='CN/')
    parser.add_argument('--MCI_dir', type=str,
                        help='subfloder of train or test dataset', default='MCI/')
    parser.add_argument('--PMCI_dir', type=str,
                        help='subfloder of train or test dataset', default='PMCI/')
    parser.add_argument('--SMCI_dir', type=str,
                        help='subfloder of train or test dataset', default='SMCI/')

    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.001)')
    parser.add_argument('--class_num', type=int, help='class_num', default=2)
    parser.add_argument('--seed', type=int, help='Seed', default=42)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='0')
    parser.add_argument('--train_root_path', type=str, help='Root path for train dataset',
                        default='/MRIbr24/train/')
    parser.add_argument('--val_root_path', type=str, help='Root path for val dataset',
                        default='MRIbr24/val/')
    parser.add_argument('--test_root_path', type=str, help='Root path for test dataset',
                        default='MRIbr24/test/')
    parser.add_argument('--batch_size', type=int, help='batch_size of data', default=4)
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=60)

    parser.add_argument("--contrastive_loss", type=bool, help='use contrastive loss or not', default=False)
    parser.add_argument("--learnable_alpha", type=bool, help='use learnable alpha or not', default=False)
    parser.add_argument("--contrastive_coefficient", type=float, help='contrastive coefficient', default=1.0)

    parser.add_argument("--oversampling", type=bool, help='oversampling or not', default=False)
    parser.add_argument("--add_weight_decay", type=bool, help='add weight decay or not', default=False)
    parser.add_argument("--weight_decay", type=float, help='weight decay', default=1e-5)

    parser.add_argument("--accumulate_loss", type=bool, help='accumalte loss or not', default=False)

    parser.add_argument("--val_test_turbo", type=bool, help='val test turbo or not', default=False)

    return parser.parse_args()
