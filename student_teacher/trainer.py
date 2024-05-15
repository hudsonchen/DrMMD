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
            os.mkdir(self.log_dir)
        
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
        elif self.args.loss=='chard':
            return CHARD(self.student,self.args.with_noise, self.args.lmbda)
    def get_optimizer(self,lr):
        if self.args.optimizer=='SGD':
            return optim.SGD(self.student.parameters(), lr=lr)

    def init_student(self,mean,std):
        weights_init_student = partial(weights_init,{'mean':mean,'std':std})
        self.student.apply(weights_init_student)

    def train(self,start_epoch=0,total_iters=0):
        wandb.login(key='c6ea42f5f183e325a719b86d84e7aed50b2dfd5c')
        wandb.init(project="student_teacher_new")
        wandb.config.update(self.args)
        print("Starting Training Loop...")
        start_time = time.time()
        
        test_mmd_all = np.zeros((self.args.total_epochs))
        test_loss_all = np.zeros((self.args.total_epochs))
        train_mmd_all = np.zeros((self.args.total_epochs))
        train_loss_all = np.zeros((self.args.total_epochs))
        train_K_norm_all = np.zeros((self.args.total_epochs))
        train_grad_K_norm_all = np.zeros((self.args.total_epochs))
        train_hess_K_norm_all = np.zeros((self.args.total_epochs))

        for epoch in tqdm(range(start_epoch, start_epoch+self.args.total_epochs)):
            total_iters,train_loss,train_mmd, train_K_norm, train_grad_K_norm, train_hess_K_norm = train_epoch(self.args, epoch,total_iters,self.loss,self.data_train,self.optimizer,'train',  device=self.device)
            total_iters,valid_loss,valid_mmd, _, _, _ = train_epoch(self.args, epoch, total_iters, self.loss,self.data_valid,self.optimizer,'valid',  device=self.device)
            
            test_loss_all[epoch] = valid_loss
            test_mmd_all[epoch] = valid_mmd
            train_loss_all[epoch] = train_loss
            train_mmd_all[epoch] = train_mmd
            train_K_norm_all[epoch] = train_K_norm
            train_grad_K_norm_all[epoch] = train_grad_K_norm
            train_hess_K_norm_all[epoch] = train_hess_K_norm

            if not np.isfinite(train_loss):
                break 

            if self.args.use_scheduler:
                self.scheduler.step(train_loss)
            if np.mod(epoch,self.args.noise_decay_freq)==0 and epoch>0:
                self.loss.student.update_noise_level()
            if np.mod(epoch,10)==0:
                new_time = time.time()
                start_time = new_time
        wandb.finish()

        print("Training Finished!")

        np.save(os.path.join(self.log_dir, 'test_mmd_all.npy'), test_mmd_all)
        np.save(os.path.join(self.log_dir, 'test_loss_all.npy'), test_loss_all)
        np.save(os.path.join(self.log_dir, 'train_mmd_all.npy'), train_mmd_all)
        np.save(os.path.join(self.log_dir, 'train_loss_all.npy'), train_loss_all)
        np.save(os.path.join(self.log_dir, 'train_K_norm_all.npy'), train_K_norm_all)
        np.save(os.path.join(self.log_dir, 'train_grad_K_norm_all.npy'), train_grad_K_norm_all)
        np.save(os.path.join(self.log_dir, 'train_hess_K_norm_all.npy'), train_hess_K_norm_all)

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

def train_epoch(args, epoch,total_iters,Loss,data_loader, optimizer,phase, device="cuda"):

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

    # Code for computing the Jacobian and Hessian of the network
    if np.mod(epoch, 1)==0:
        if phase == 'train':
            batch_size = inputs.shape[0]
            out_ = Loss.student(inputs)[:, 0, :]
            K = (out_.T @ out_) / batch_size
            K_norm = K.mean()

            J = []
            for i in range(args.num_particles):
                Loss.student.zero_grad()
                out = Loss.student(inputs)[:, 0, i]
                dummy_out = tr.ones_like(out)
                out.backward(dummy_out, retain_graph=True)
                J.append(torch.cat([p.grad.view(-1) for p in Loss.student.parameters()]))
            Jacobian = torch.stack(J)
            grad_K = (Jacobian.T @ out_.mean(0))
            grad_K_norm = (grad_K ** 2).mean()
            
            H = []
            for i in range(args.num_particles):
                output = Loss.student(inputs)[:, 0, i].sum()
                hessian = []
                for idx, param in enumerate(Loss.student.parameters()):
                    grad_param = torch.autograd.grad(output, param, create_graph=True)[0]
                    for jdx, param2 in enumerate(Loss.student.parameters()):
                        second_order_grad = torch.autograd.grad(grad_param.sum(), param2, retain_graph=True)[0]
                        hessian.append(second_order_grad.view(-1))
                hessian = torch.concatenate(hessian)
                H.append(hessian)
            Hess_K = torch.stack(H)
            grad_K = (Hess_K.T @ out_.mean(0))
            Hess_K_norm = (Hess_K ** 2).mean()
            pause = True
        else:
            K_norm = 0
            grad_K_norm = 0
            Hess_K_norm = 0
    ###
            
    if np.mod(epoch, 100)==0:
        if phase=='valid':
            wandb.log({"Validation Loss": total_loss, "Validation MMD": total_mmd}, step=epoch)
        elif phase=='train':
            wandb.log({"Train Loss": total_loss, "Train MMD": total_mmd}, step=epoch)
            wandb.log({"Validation K_norm": K_norm, "Validation grad_K_norm": grad_K_norm, "Validation Hess_K_norm": Hess_K_norm}, step=epoch)
        else:
            pass
    return total_iters, total_loss, total_mmd, K_norm, grad_K_norm, Hess_K_norm
