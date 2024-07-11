import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=28250)
parser.add_argument('--epochs',type=int,default=*)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=True)
parser.add_argument('--eval_step',type=int,default=*)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default=r'**')
parser.add_argument('--model_direpoch',type=str,default=r'**')
parser.add_argument('--best_model_dir',type=str,default=r'**')
# parser.add_argument('--best_model_dir',type=str,default='./trained_models1/')
parser.add_argument('--trainset',type=str,default='ots_train')
parser.add_argument('--testset',type=str,default='ots_test')
parser.add_argument('--net',type=str, default='Dehaze')
# parser.add_argument('--gps',type=int,default=3,help='residual_groups')
# parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=32,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=256,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')
parser.add_argument('--model_name', type=str,default='**')
parser.add_argument('--pre_model', type=str,default='null')
parser.add_argument('--transfer', type=bool,default=False)
parser.add_argument('--choose',type=int,default=218,help='choose different models ')
parser.add_argument('--w_loss_l1', type=float, default=1.0)
parser.add_argument('--w_loss_vgg7', type=float, default=0.04)
parser.add_argument('--w_loss_vgg', type=float, default=0.04)
parser.add_argument('--w_loss_lap', type=float, default=0)
# parser.add_argument('--w_loss_vgg', type=float, default=0)
parser.add_argument('--w_loss_infonce', type=float, default=0)
parser.add_argument('--w_loss_npair', type=float, default=0)
parser.add_argument('--w_loss_triplet', type=float, default=0)
parser.add_argument('--w_loss_vgg2', type=float, default=0)
parser.add_argument('--w_loss_vgg3', type=float, default=0)
parser.add_argument('--w_loss_vgg4', type=float, default=0)
# parser.add_argument('--w_loss_vgg7', type=float, default=0.4)
parser.add_argument('--is_ab', type=bool, default=False)
parser.add_argument('--w_loss_fft', type=float, default=0)
parser.add_argument('--w_loss_conf', type=float, default=0)
parser.add_argument('--w_loss_rfft', type=float, default=0)
parser.add_argument('--w_loss_consis', type=float, default=0)
parser.add_argument('--w_loss_dis', type=float, default=0)
parser.add_argument('--w_loss_proto', type=float, default=0)
parser.add_argument('--w_CR_ex', type=float, default=0)
parser.add_argument('--w_loss_adv', type=float, default=0)
parser.add_argument('--w_loss_causa', type=float, default=0)
parser.add_argument('--w_loss_fea', type=float, default=0)
parser.add_argument('--w_loss_fea_in', type=float, default=0)

parser.add_argument('--w1', type=float, default=0)
parser.add_argument('--w2', type=float, default=0)

parser.add_argument('--pre_train_epochs', type=int, default=10, help='train with l1 and fft')

parser.add_argument('--lr_decay', type=bool, default=True)
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='lr decay rate')
parser.add_argument('--lr_decay_win', type=int, default=4, help='lr decay windows: epoch')
parser.add_argument('--eval_dataset', type=bool, default=False)


parser.add_argument("--iters_balance", type=int, default=20000)
parser.add_argument("--decorrelation_type", choices=["global", "group"], default="global")
parser.add_argument("--seed", type=int, default=3)
##因果
# parser.add_argument('--lambda_decay_rate', type=float, default=1, help='ratio of epoch for lambda to decay')
# parser.add_argument('--lambda_decay_epoch', type=int, default=5, help='number of epoch for lambda to decay')
# parser.add_argument('--min_lambda_times', type=float, default=0.01, help='number of global table levels')
# parser.add_argument('--lambdapre', type=float, default=1, help='weight for pre_weight1 ')
# parser.add_argument('--lambdap', type=float, default=70.0, help='weight decay for weight1 ')
# parser.add_argument('--decay_pow', type=float, default=2, help='value of pow for weight decay')
# parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
# parser.add_argument('--num_f', type=int, default=1, help='number of fourier spaces')
# parser.add_argument('--lrbl', type=float, default=1.0, help='learning rate of balance')
# # optimization
# # parser.add_argument('--resume', action='store_true', default=False, help='continue training the model')
# parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
# parser.add_argument('--momentum', default=0.99, type=float, help='Momentum parameter used in the Optimizer.')
# parser.add_argument('--epochs', default=100, type=int, help='nutraining batch size')

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not opt.transfer:
	opt.model_name = opt.trainset + '_' + opt.net.split('.')[0] +'_' + str(opt.choose) + '_'+str(opt.crop_size) + '_' + str(opt.model_name)
else:
	opt.model_name = 'ots_train_ffa_3_19_pretrain'

opt.model_dir1 = opt.model_dir + opt.model_name + '.pk'


log_dir = 'logs/'+opt.model_name if not opt.transfer else 'logs/'+opt.model_name+'_transfer_' + opt.model_info
# model_name = opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.crop_size)
# opt.model_dir = opt.model_dir+model_name+'.pk'
# log_dir='logs/'+model_name

print(opt)
print('model_dir:',opt.model_dir)
print(f'log_dir: {log_dir}')


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{opt.model_name}"):
	os.mkdir(f'samples/{opt.model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
