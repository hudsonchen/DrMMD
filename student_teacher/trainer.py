from __future__ import print_function
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import wandb
from tqdm import tqdm
import numpy as np
from functools import partial
import sys,os,time,itertools

from networks import *
from losses import *

class Trainer(object):
    def __init__(self,args):
        self.args = args
        self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() and args.device>-1 else 'cpu'
        self.dtype = get_dtype(args)
        self.log_dir = os.path.join(args.log_dir, args.log_name+'_loss_' + args.loss +'_noise_level_'+str(args.noise_level)+'_lmbda_'+ str(args.lmbda))
        self.log_dir += f"_lr_{str(args.lr)}_with_noise_{str(args.with_noise)}_noise_decay_freq_{args.noise_decay_freq}"
        if not os.path.isdir(self.log_dir):
            print(os.getcwd())
            os.makedirs(self.log_dir, exist_ok=True)
        if args.log_in_file:
            self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
            sys.stdout = self.log_file
        print('==> Building model..')
        self.build_model()


    def build_model(self):
        torch.manual_seed(self.args.seed)
        if not self.args.with_noise:
            self.args.noise_level = 0.
        self.teacherNet = get_net(self.args,self.dtype,self.device,'teacher')
        self.student = get_net(self.args,self.dtype,self.device,'student')
        self.data_train = get_data_gen(self.teacherNet,self.args,self.dtype,self.device)
        self.data_valid = get_data_gen(self.teacherNet,self.args,self.dtype,self.device)
        
        self.loss = self.get_loss()

        self.optimizer = self.get_optimizer(self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience = 50,verbose=True, factor = 0.9)
        #self.get_reg_dist()

    def get_loss(self):
        if self.args.loss=='mmd_noise_injection':
            return MMD(self.student,self.args.with_noise)
        elif self.args.loss=='mmd_diffusion':
            return MMD_Diffusion(self.student)
        elif self.args.loss=='sobolev':
            return Sobolev(self.student)
        elif self.args.loss=='drmmd':
            return drmmd(self.student,self.args.with_noise, self.args.lmbda)
    def get_optimizer(self,lr):
        if self.args.optimizer=='SGD':
            return optim.SGD(self.student.parameters(), lr=lr)

    def init_student(self,mean,std):
        weights_init_student = partial(weights_init,{'mean':mean,'std':std})
        self.student.apply(weights_init_student)

    def train(self,start_epoch=0,total_iters=0):
        # wandb.login(key='c6ea42f5f183e325a719b86d84e7aed50b2dfd5c')
        # wandb.init(project="student_teacher_new")
        # wandb.config.update(self.args)
        print("Starting Training Loop...")
        start_time = time.time()
        
        test_mmd_all = np.zeros((self.args.total_epochs))
        test_loss_all = np.zeros((self.args.total_epochs))
        train_mmd_all = np.zeros((self.args.total_epochs))
        train_loss_all = np.zeros((self.args.total_epochs))

        for epoch in tqdm(range(start_epoch, start_epoch+self.args.total_epochs)):
            total_iters,train_loss,train_mmd = train_epoch(epoch,total_iters,self.loss,self.data_train,self.optimizer,'train',  device=self.device)
            total_iters,valid_loss,valid_mmd = train_epoch(epoch, total_iters, self.loss,self.data_valid,self.optimizer,'valid',  device=self.device)
            
            test_loss_all[epoch] = valid_loss
            test_mmd_all[epoch] = valid_mmd
            train_loss_all[epoch] = train_loss
            train_mmd_all[epoch] = train_mmd

            if not np.isfinite(train_loss):
                break 

            if self.args.use_scheduler:
                self.scheduler.step(train_loss)
            if np.mod(epoch,self.args.noise_decay_freq)==0 and epoch>0:
                self.loss.student.update_noise_level()
            if np.mod(epoch,10)==0:
                new_time = time.time()
                start_time = new_time
        
        print("Training Finished!")
        # wandb.finish()
        np.save(os.path.join(self.log_dir, 'test_mmd_all.npy'), test_mmd_all)
        np.save(os.path.join(self.log_dir, 'test_loss_all.npy'), test_loss_all)
        np.save(os.path.join(self.log_dir, 'train_mmd_all.npy'), train_mmd_all)
        np.save(os.path.join(self.log_dir, 'train_loss_all.npy'), train_loss_all)
    
        return train_loss,valid_loss,train_mmd,valid_mmd


def get_data_gen(net,args,dtype,device):
    params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0}
    if args.input_data=='Spherical':
        teacher  = SphericalTeacher(net,args.N_train,dtype,device)
    return data.DataLoader(teacher, **params)

def get_net(args,dtype,device,net_type):
    non_linearity = quadexp()
    if net_type=='teacher':
        weights_init_net = partial(weights_init,{'mean':args.mean_teacher,'std':args.std_teacher})
        if args.teacher_net=='OneHidden':
            Net = OneHiddenLayer(args.d_int,args.H,args.d_out,non_linearity = non_linearity,bias=args.bias)
    if net_type=='student':
        weights_init_net = partial(weights_init,{'mean':args.mean_student,'std':args.std_student})
        if args.student_net=='NoisyOneHidden':
            Net = NoisyOneHiddenLayer(args.d_int, args.H, args.d_out, args.num_particles, non_linearity = non_linearity, noise_level = args.noise_level,noise_decay=args.noise_decay,bias=args.bias)

    Net.to(device)
    if args.dtype=='float64':
        Net.double()
    
    Net.apply(weights_init_net)
    return Net

def get_dtype(args):
    if args.dtype=='float32':
        return torch.float32
    else:
        return torch.float64


def weights_init(args,m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=args['mean'],std=args['std'])
        if m.bias:
            m.bias.data.normal_(mean=args['mean'],std=args['std'])

def train_epoch(epoch,total_iters,Loss,data_loader, optimizer,phase, device="cuda"):

    # Training Loop
    # Lists to keep track of progress

    if phase == 'train':
        Loss.student.train(True)  # Set model to training mode
    else:
        Loss.student.train(False)  # Set model to evaluate mode
    
    cum_loss = 0
    cum_mmd = 0
    # For each epoch

    # For each batch in the dataloader
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if phase=="train":
            out = tr.mean(Loss.student(inputs),dim = -1).clone().detach()
            cum_mmd += 0.5*tr.mean((targets - out)**2).item()

            total_iters += 1
            Loss.student.zero_grad()
            loss = Loss(inputs, targets)
            # Calculate the gradients for this batch
            loss.backward()
            optimizer.step()
            loss = loss.item()
            cum_loss += loss

        elif phase=='valid':
            loss = Loss(inputs, targets).item()
            cum_loss += loss
            out = tr.mean(Loss.student(inputs),dim = -1).clone().detach()
            cum_mmd += 0.5*tr.mean((targets - out)**2).item()

    total_loss = cum_loss/(batch_idx+1)
    total_mmd = cum_mmd/(batch_idx+1)
    # if np.mod(epoch, 100)==0:
    #     if phase=='valid':
    #         wandb.log({"Validation Loss": total_loss, "Validation MMD": total_mmd}, step=epoch)
    #     elif phase=='train':
    #         wandb.log({"Train Loss": total_loss, "Train MMD": total_mmd}, step=epoch)
    #     else:
    #         pass
    if np.mod(epoch, 100)==0:
        print('Epoch: '+ str(epoch) + ' | ' + phase + ' mmd: ' + str(round(total_mmd, 6)) + ' loss: ' + str(round(total_loss, 6)))
        # pass
    return total_iters, total_loss, total_mmd