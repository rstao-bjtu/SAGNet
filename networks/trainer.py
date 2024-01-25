import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.delr = opt.delr
        self.lamda = opt.lamda

#        if self.isTrain and not opt.continue_train:
#            self.model = resnet50(pretrained=True, lnum=opt.lnum)
            # self.model.fc = nn.Linear(2048, 1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        self.model = get_model('CLIP:ViT-L/14')
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        params = []
        for name, p in self.model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                params.append(p)
            elif name.startswith("Domain_classifier"):
                params.append(p)
            elif name.startswith("model.visual.transformer.resblocks.23"):
                params.append(p)
            # elif name.startswith("model.visual.transformer.resblocks.22"):
            #     params.append(p)
            # elif name.startswith("model.visual.transformer.resblocks.21"):
            #     params.append(p)
            elif name=="model.visual.ln_post.weight" or name=="model.visual.ln_post.bias":
                params.append(p)
            else:
                p.requires_grad = False
                # p.requires_grad = True


#        if not self.isTrain or opt.continue_train:
#            self.model = resnet50(num_classes=1)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if pa.requires_grad: print('='*20, 'requires_grad Ture',name)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if not pa.requires_grad: print('='*20, 'requires_grad False',name)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if pa.requires_grad: print('='*20, 'requires_grad Ture',name)
        print('='*30)
        net_params = sum(map(lambda x: x.numel(), self.model.model.visual.parameters())) +  sum(map(lambda x: x.numel(), self.model.Domain_classifier.parameters())) + sum(map(lambda x: x.numel(), self.model.fc.parameters()))

        print(f'Model parameters {net_params:,d}')

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                # self.optimizer = torch.optim.Adam(self.model.parameters(),
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        # self.model.to(opt.gpu_ids[0])
        self.model = nn.DataParallel(self.model).cuda()

 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.delr
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/self.delr} to {param_group["lr"]} with delr {self.delr}')
        print('*'*25)
        return True

    def set_input(self, input_d1, input_d2):
        self.input_d1 = input_d1[0].to(self.device)
        self.input_d2 = input_d2[0].to(self.device)
        self.label_d1 = input_d1[1].to(self.device).float()
        self.label_d2 = input_d2[1].to(self.device).float()


    def forward_d1(self):
        self.output_d1_cls, self.output_d1_domain = self.model.module.forward_train(self.input_d1, self.label_d1)
    
    def forward_d2(self):
        self.output_d2_cls, self.output_d2_domain = self.model.module.forward_train(self.input_d2, self.label_d2)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters_baseline(self):
        self.forward_d1()
        self.forward_d2()
        # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss_d1_cls = self.loss_fn(self.output_d1_cls.squeeze(1), self.label_d1)
        # self.loss_d1_domain = self.loss_fn(self.output_d1_domain.squeeze(1), torch.zeros_like(self.output_d1_domain.squeeze(1)).float().cuda())
        self.loss_d2_cls = self.loss_fn(self.output_d2_cls.squeeze(1), self.label_d2)
        # self.loss_d2_domain = self.loss_fn(self.output_d2_domain.squeeze(1), torch.ones_like(self.output_d2_domain.squeeze(1)).float().cuda())
        self.loss = self.loss_d1_cls + self.loss_d2_cls
        print("loss_d1_cls: {} loss_d2_cls: {}".format(self.loss_d1_cls, self.loss_d2_cls))

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def optimize_parameters_ours(self):
        self.forward_d1()
        self.forward_d2()
        # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss_d1_cls = self.loss_fn(self.output_d1_cls.squeeze(1), self.label_d1)
        self.loss_d1_domain = self.loss_fn(self.output_d1_domain.squeeze(1), torch.zeros_like(self.output_d1_domain.squeeze(1)).float().cuda())
        self.loss_d2_cls = self.loss_fn(self.output_d2_cls.squeeze(1), self.label_d2)
        self.loss_d2_domain = self.loss_fn(self.output_d2_domain.squeeze(1), torch.ones_like(self.output_d2_domain.squeeze(1)).float().cuda())

        self.loss = self.loss_d1_cls + self.lamda*self.loss_d1_domain + self.loss_d2_cls + self.lamda*self.loss_d2_domain
        print("loss_d1_cls: {} loss_d1_domain: {} loss_d2_cls: {} loss_d2_domain: {}".format(self.loss_d1_cls, self.loss_d1_domain, self.loss_d2_cls, self.loss_d2_domain))

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

