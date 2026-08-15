import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging',
                        default='Exp1')

    parser.add_argument('--std',
                        type=float,
                        help='standard deviation for gaussian distribution learning',
                        default=0.60)

    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=True)

    parser.add_argument('--type',
                        type=str,
                        help='type of the model: USDL or MUSDL',
                        choices=['USDL', 'MUSDL'],
                        default='USDL')

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate for projector/decoder/scorer',
                        default=2e-5)

    parser.add_argument('--backbone_lr',
                        type=float,
                        help='learning rate for trainable I3D backbone layers',
                        default=1e-6)

    parser.add_argument('--eta_min',
                        type=float,
                        help='minimum learning rate for cosine scheduler',
                        default=5e-7)

    parser.add_argument('--hard_label_weight',
                        type=float,
                        help='auxiliary hard-label CE weight',
                        default=0.06)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-3)

    parser.add_argument('--temporal_aug',
                        type=int,
                        help='the maximum of random temporal shift, ranges from 0 to 6',
                        default=4)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=3407)

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=8)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='2,3')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=16)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=16)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=100)
    

    return parser



