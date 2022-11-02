
import torch
import utils
import numpy as np
from ControlPanel import ControlPanel

from torch import nn
from copy import deepcopy
from .nn import NPCModule,NetStatistics
from statistics import mean
from NPCLogger import NPCLog
from torch.utils.data import DataLoader

class LogisticRegression(NPCModule):
    def __init__(self,input_dim = 768,cls_num = 10,name = ""):
        NPCModule.__init__(self,name)
        self.linear = nn.Linear(input_dim,cls_num)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def single_output_to_label(o):
        return o > 0.5

    def forward(self,x):
        x = torch.flatten(x,1)
        return self.sigmoid(self.linear(x))


class TinyIMGCNN(NPCModule):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        NPCModule.__init__(self)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_features,32,kernel_size=5,padding=0,stride=1,bias=True),
            torch.nn.ReLU(inplace=True), 
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=5,padding=0,stride=1,bias=True),
            torch.nn.ReLU(inplace=True), 
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out

from torch.autograd import Variable
from numpy import random as nprandom

class MinistGAN(NPCModule):
    def __init__(self, name='',img_flatten_size = 784):
        NPCModule.__init__(self,name)
        self.hiddendim = 100
        self.img_flatten_size = img_flatten_size
        self.discriminator = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(img_flatten_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Softmax(1),
            )

        self.generator = nn.Sequential(
                nn.Linear(self.hiddendim,512),
                nn.Tanh(),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, img_flatten_size),
                nn.Tanh()
            )

    def toImg(self,genRes):
        flatten_imgs = genRes
        bs,_ = flatten_imgs.size()
        imgs = torch.zeros((bs,28,28))
        for b in range(bs):
            for line in range(0,self.img_flatten_size-28,28):
                imgs[b,line // 28,:] = flatten_imgs[b,line:line+28]
            # imgs[b,:,:] = imgs[b,:,:].norm()
        return imgs

    def GeneratorLast(self,z = None,batch_size = 1,auto_sample = True):
        if auto_sample:
            self.lastz = deepcopy()
            z = Variable(torch.Tensor(nprandom.normal(0, 1, (batch_size, self.hiddendim))))
            return self.generator(z)
        else:
            return self.generator(z)

    def Generator(self,z = None,batch_size = 1,auto_sample = True,sort_sample = False):
        if auto_sample:
            sample = nprandom.normal(0, 1, (batch_size, self.hiddendim))
            if sort_sample:
                sample = np.array(sorted(sample,key = lambda a:np.sqrt((a*a).sum())))
            z = Variable(torch.Tensor(sample))
            return self.generator(z)
        else:
            return self.generator(z)

    def Discriminator(self,x):
        return self.discriminator(x)

    def Train(net, dataset,echo=20, lr=0.001, endACU=99.5, batch_echo=100, save=True, save_acu_lower_bound=80):
        testdataset = None
        _,device,Loss,prgl = net.__initTrain__(dataset,testdataset,lr=lr)

        loader = DataLoader(dataset, batch_size = ControlPanel.batch_size, shuffle=True,collate_fn=dataset.collate_func)

        net.train_mean_loss,acu,saved,genacu = [],{True:0,False:0},False,{True:0,False:0}
        flattenImg = torch.nn.Flatten(1)
        dicacuv,genacuv = 2,1
                    
        for echo_count in range(echo):
            opt_G = torch.optim.Adam(net.generator.parameters(), lr = lr * (4 if dicacuv > genacuv else 1),betas=(0.5,0.99))
            opt_D = torch.optim.Adam(net.discriminator.parameters(), lr = lr * (4 if dicacuv < genacuv else 1),betas=(0.5,0.99))
            all_batch_losses,all_batch_gen_losses = [],[]
            for batch_count, (imgs, _) in enumerate(loader):
                # Optimizers
                imgs = imgs.to(device)
                
                # Adversarial ground truths
                true_label = torch.ones((imgs.size(0)),dtype=torch.int64)
                fake_label = torch.zeros((imgs.size(0)),dtype=torch.int64)
                # Loss measures generator's ability to fool the discriminator
                
                opt_D.zero_grad()
                opt_G.zero_grad()
                gen_imgs = net.Generator(batch_size = imgs.size(0))
                go = net.Discriminator(gen_imgs)
                # print(go,true_label)
                # input()
                genacu = utils.Counter(net.train_cmp(go,true_label).tolist(),genacu)
                    
                # utils.Counter(net.train_cmp(go,valid).tolist()[0])
                g_loss = Loss(go,true_label)
                g_loss.backward()
                opt_G.step()
                all_batch_gen_losses.append(torch.mean(g_loss.detach()).item())

                # Measure discriminator's ability to classify real from generated samples
                rfimg = torch.cat([flattenImg(imgs),gen_imgs.detach()],dim=0)
                rfimg_label = torch.cat([true_label,fake_label],dim=0)
                shffle = torch.randperm(rfimg.size(0))
                rfimg = rfimg[shffle,:]
                rfimg_label = rfimg_label[shffle]
                opt_D.zero_grad()
                do = net.Discriminator(rfimg)
                d_loss = Loss(do,rfimg_label)
                d_loss.backward()
                opt_D.step()

                acu = utils.Counter(net.train_cmp(do,rfimg_label).tolist(),acu)
                all_batch_losses.append(torch.mean(d_loss.detach()).item())
                if batch_count % batch_echo == 0:
                    dicacuv = net.PrintBatchInfo(acu,echo_count,batch_count,prgl,all_batch_losses)
                    genacuv = net.PrintBatchInfo(genacu,echo_count,batch_count,prgl,all_batch_gen_losses)
                    if genacu[True]+genacu[False] > 1000 and genacuv > 80:
                        net.__SaveBest__(100*genacu[True]/(genacu[True]+genacu[False]))
                    acu,genacu = {True:0,False:0},{True:0,False:0}
            #statist
            net.train_mean_loss.append(mean(all_batch_losses))
            acuv = -1
            if (acu[True]+acu[False]) != 0:
                acuvalue = net.PrintBatchInfo(acu,echo_count,batch_count,prgl,all_batch_losses)
                genacuv = net.PrintBatchInfo(genacu,echo_count,batch_count,prgl,all_batch_gen_losses)
                acu,genacu = {True:0,False:0},{True:0,False:0}

            ###test or eval net
            if testdataset != None:
                acuv = net.Test(testdataset,False)

            ###save the best net
            if acuv > save_acu_lower_bound and acuv > net.bestACU and save:
                net.bestACU = acuv
                net.save("%s"%(net.name))
                NPCLog("There exists the best net of test acu...saved.")
                saved = True

            ##early stop
            if (mean(all_batch_losses) < 1e-6 or acuv > endACU) and saved:
                NPCLog("acu statified setting of user,program stop training the model.")
                break

        NPCLog(title="")
        if saved:
            return utils.CheckModel(net.name)
        elif save:
            return utils.CheckModel(net.name,lambda:net)
        else:
            return net