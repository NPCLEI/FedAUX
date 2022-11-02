
from torch.autograd import Variable
from numpy import random as nprandom
from statistics import mean

from NPCLogger import NPCLog
from .nn import NPCModule
from torch import nn
import torch
import ControlPanel
from torch.utils.data import DataLoader
import utils

class MinistGAN:
    class Discriminator(NPCModule):
        def __init__(self,img_flatten_size = 784, name=''):
            super().__init__(name)
            self.discriminator = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(img_flatten_size, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self,x):
            return self.discriminator(x)

    class Generator(NPCModule):
        def __init__(self,img_flatten_size = 784, name=''):
            super().__init__(name)
            self.generator = nn.Sequential(
                    nn.Linear(self.hiddendim,1024),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(1024, 1024),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(1024, img_flatten_size),
                    nn.Tanh()
                )

        def forward(self,z = None,batch_size = 1,auto_sample = True):
            if auto_sample:
                z = Variable(torch.Tensor(nprandom.normal(0, 1, (batch_size, self.hiddendim))))
                return self.generator(z)
            else:
                return self.generator(z)

    def __init__(self, name='',img_flatten_size = 784):
        NPCModule.__init__(self,name)
        self.hiddendim = 100
        self.img_flatten_size = img_flatten_size
        self.generator = MinistGAN.Generator()
        self.discriminator = MinistGAN.Discriminator()

    def toImg(self,genRes):
        flatten_imgs = genRes
        bs,_ = flatten_imgs.size()
        imgs = torch.zeros((bs,28,28))
        for b in range(bs):
            for line in range(0,self.img_flatten_size-28,28):
                imgs[b,line // 28,:] = flatten_imgs[b,line:line+28]
            # imgs[b,:,:] = imgs[b,:,:].norm()
        return imgs

    def GenerateImg(self,z = None,batch_size = 1,auto_sample = True):
        return self.generator(z,batch_size,auto_sample)

    def Train(net, dataset,echo=20, lr=0.001, endACU=99.5, batch_echo=100, save=True, save_acu_lower_bound=80):
        testdataset = None
        _,device,Loss,prgl = net.initTrain(dataset,testdataset,lr=lr)

        loader = DataLoader(dataset, batch_size = ControlPanel.batch_size, shuffle=True,collate_fn=dataset.collate_func)

        # Optimizers
        optimizer_G = torch.optim.Adam(net.generator.parameters(), lr = lr,betas=(0.5,0.99))
        optimizer_D = torch.optim.Adam(net.discriminator.parameters(), lr = lr,betas=(0.5,0.99))

        net.train_mean_loss,acu,saved,genacu = [],{True:0,False:0},False,{True:0,False:0}
        flattenImg = torch.nn.Flatten(1)

        for echo_count in range(echo):
            all_batch_losses = []
            for batch_count, (imgs, _) in enumerate(loader):
                imgs = imgs.to(device)
                
                # Adversarial ground truths
                true_label = torch.Tensor(imgs.size(0), 1).fill_(1.0)
                fake_label = torch.Tensor(imgs.size(0), 1).fill_(0.0)
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = net.Generator(batch_size = imgs.size(0))
                # Loss measures generator's ability to fool the discriminator
                go = net.Discriminator(gen_imgs)
                genacu = utils.Counter(net.train_cmp(go,true_label).tolist()[0],genacu)
                g_loss = Loss(go, true_label)
                # utils.Counter(net.train_cmp(go,valid).tolist()[0])

                g_loss.backward()
                optimizer_G.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                rfimg = torch.cat([flattenImg(imgs),gen_imgs.detach()],dim=0)
                rfimg_label = torch.cat([true_label,fake_label],dim=0)
                shffle = torch.randperm(rfimg.size(0))
                rfimg = rfimg[shffle,:]
                rfimg_label = rfimg_label[shffle,:]

                do = net.Discriminator(rfimg)
                loss = Loss(do,rfimg_label)

                loss.backward()
                optimizer_D.step()

                acu = utils.Counter(net.train_cmp(do,rfimg_label).tolist()[0],acu)
                all_batch_losses.append(torch.mean(loss).item())
                if batch_count % batch_echo == 0:
                    acuvalue = net.PrintBatchInfo(acu,echo_count,batch_count,prgl,all_batch_losses)
                    NPCLog(genacu)
                    acu,genacu = {True:0,False:0},{True:0,False:0}
            #statist
            net.train_mean_loss.append(mean(all_batch_losses))
            acuv = -1
            if (acu[True]+acu[False]) != 0:
                acuvalue = net.PrintBatchInfo(acu,echo_count,batch_count,prgl,all_batch_losses)
                NPCLog(genacu)
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