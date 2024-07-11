import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from CGID import  fusion_net
# from FSDA_model3 import Net
from SRDO import SRDO
# import models
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
from reweighting import weight_learner
import json
from sssim_loss import msssim
# from coral import coral
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
# from option import opt,model_name,log_dir
from data_utils2 import *
from PerceptualLoss import LossNetwork as PerLoss
from CR import *
from torchvision.models import vgg16
models_={
    'Dehaze': fusion_net(),
}
loaders_={
    # 'its_train':ITS_train_loader,
    # 'its_test':ITS_test_loader,
    'ots_train':OTS_train_loader,
    'ots_test':OTS_test_loader
}
start_time=time.time()
model_name = opt.model_name
steps = opt.eval_step * opt.epochs
T = steps
criterion3 = nn.L1Loss(reduce=False).cuda()
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr
# args = parser.parse_args()
def train(net,loader_train,loader_test,optim,criterion):
    losses=[]
    llossl1=[]
    ssimloss=[]
    llosscont =[]
    llossper=[]
    lloss_C=[]
    start_step=0
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
    print(os.path.exists(opt.model_dir1))
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(opt.device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    Per_loss = PerLoss(vgg_model).to(opt.device)
    Per_loss.eval()
    if opt.resume and os.path.exists(opt.model_dir1):
        if opt.pre_model !='null':
            ckp = torch.load('./trained_models/' + opt.pre_model)
        else:
            ckp = torch.load(opt.model_dir1)
        print(f'resume from {opt.model_dir}')
        # ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        # llossper = ckp['loss_per']
        lloss_C = ckp['lloss_C']
        llosscont = ckp['llosscont']
        net.load_state_dict(ckp['model'])
        # optim.load_state_dict(ckp['optimizer'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs= ckp['psnrs']
        ssims= ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else :
        print('train from scratch *** ')
    for step in range(start_step+1, steps+1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr=lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        b,c,h,w=x.shape
        y = y.to(opt.device)
        out,z,z1 = net(x)
        bb,cc,hh,ww=z.size()
        zz=z.reshape(bb,cc*hh*ww)
        bb1, cc1, hh1, ww1 = z1.size()
        zz1 = z1.reshape(bb1, cc1 * hh1 * ww1)
        W = CQZ(z, 3, decorrelation_type=opt.decorrelation_type, max_iter=opt.iters_balance,
                 random_state=opt.seed)
        W=torch.from_numpy(W).cuda().to(torch.float32)
        mseloss = torch.sum(criterion3(out, y), (1, 2, 3)) / (c * h * w)
        cc=mseloss.view(1, -1)
        loss_C = mseloss.view(1, -1).mm(W).view(1)
        # pre_features = torch.zeros(opt.bs, 512).cuda()
        # pre_weight1 = torch.ones(opt.bs, 1).cuda()
        # weight1, pre_features, pre_weight1 = weight_learner(z, pre_features, pre_weight1, args, step, opt.bs)
        # pre_features.data.copy_(pre_features)
        # pre_weight1.data.copy_(pre_weight1)
        # mseloss = torch.sum(criterion3(out, y), (1, 2, 3)) / (c*h*w)
        # loss_C = mseloss.view(1, -1).mm(weight1).view(1)
        # print(loss_C)
        loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0
        # loss = []
        # loss_per=0
        if opt.w_loss_l1 > 0:
            loss_rec = criterion[0](out, y)
        if opt.w_loss_vgg7 > 0:
            # loss_vgg7, all_ap, all_an = criterion[1](out, y, x)
            loss_ct = criterion[1](out, y, x)
        # loss = opt.w_loss_l1 * loss_rec + opt.w_loss_vgg7 * loss_ct
        # loss=criterion[0](out,y)
        # if opt.perloss:
        # 	loss_per1 = criterion[1](out,y)
        msssim_loss = msssim
        ssim_loss = 1 - msssim_loss(out, y, normalize=True)
        # loss_per =  coral(zz, zz1)
        loss_per = criterion[0](zz, zz1)
        lossl1 = opt.w_loss_l1 * loss_rec
        losscont = opt.w_loss_vgg7 * loss_ct
        loss = losscont+loss_C+loss_per+0.5*ssim_loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        ssimloss.append((ssim_loss.item()))
        llossl1.append(lossl1.item())
        lloss_C.append(loss_C.item())
        # llossper.append(loss_per.item())
        llosscont.append((losscont.item()))
        # print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

        print(f'\rlosses:{loss.item():.5f} ssimloss:{0.5*ssim_loss:.5f} lossper:{loss_per:.5f}  Lw:{loss_C.item():.5f}  losscont: {losscont :.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
        end='', flush=True)
        #with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        #	writer.add_scalar('data/loss',loss,step)

        if step % opt.eval_step == 0:
            epoch = int(step / opt.eval_step)
            save_model_dir = opt.model_dir
            with torch.no_grad():
                ssim_eval,psnr_eval = yuce(net,loader_test, max_psnr,max_ssim,step)
            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'
            print(log)
            with open(f'./logs_train/{opt.model_name}.txt', 'a') as f:
                f.write(log + '\n')

            # ckp_path = os.path.join(model_dir, 'model-' + str(epoch + 1 + restart_epoch) + 'epoch')
            # torch.save(model.state_dict(), ckp_path)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            opt.model_direpoch1 = opt.model_direpoch + opt.model_name
            epoch_model_dir =opt.model_direpoch1+'_'+str(epoch) + 'epoch' + '.pk'
            print(f'\n model saved at step :{step}| epoch: {epoch} | ssim_eval:{ssim_eval:.4f}| psnr_eval:{psnr_eval:.4f}')
            torch.save({
                'epoch': epoch,
                'step': step,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'llossl1': llossl1,
                'lloss_C': lloss_C,
                'llossper':  llossper,
                'llosscont': llosscont,
                'model': net.state_dict()
            }, epoch_model_dir)
            if ssim_eval >= max_ssim and psnr_eval >= max_psnr :
                max_ssim=max(max_ssim,ssim_eval)
                max_psnr=max(max_psnr,psnr_eval)
                # model_dir = opt.model_dir + '.pk'
                print(f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                torch.save({
                            'epoch': epoch,
                            'step': step,
                            'max_psnr': max_psnr,
                            'max_ssim': max_ssim,
                            'ssims': ssims,
                            'psnrs': psnrs,
                            'losses': losses,
                            'llossper': llossper,
                            'lloss_C': lloss_C,
                            'llosscont': llosscont,
                            'model': net.state_dict()
                },opt.model_dir1)
                # print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
    np.save(f'./numpy_files/{model_name}_{opt.steps}_llossl1_**.npy', llossl1)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_llosscont_**.npy',llosscont)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_llossC_**.npy', lloss_C)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses_**.npy',losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_llossper_**.npy', llossper)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims_**.npy',ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs_**.npy',psnrs)

def yuce(net,loader_test,max_psnr,max_ssim,step):
    net.eval()
    torch.cuda.empty_cache()
    ssims=[]
    psnrs=[]
    #s=True
    for i ,(inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device); targets = targets.to(opt.device)
        with torch.no_grad():
            pred,z,z1 = net(inputs)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        #if (psnr1>max_psnr or ssim1 > max_ssim) and s :
        #		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
        #		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
        #		s=False
    return np.mean(ssims), np.mean(psnrs)

def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    set_seed_torch(666)
    if not opt.resume and os.path.exists(f'./logs_train/{opt.model_name}.txt'):
        print(f'./logs_train/{opt.model_name}.txt 已存在，请删除该文件……')
        exit()

    with open(f'./logs_train/args_{opt.model_name}.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=1)
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark=True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=opt.is_ab))

            # criterion.append(PerLoss(vgg_model).to(opt.device))
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    train(net,loader_train,loader_test,optimizer,criterion)


