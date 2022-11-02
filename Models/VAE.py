import torch
from torch import layer_norm, nn
from torch.nn import functional as F

class VAE(nn.Module):

    @staticmethod
    def CreateMLP(dims=[200,20,2]):
        res = nn.ModuleList()
        lastOutput = dims[0]
        for dim in dims[1:]:
            res.append(torch.nn.Linear(lastOutput, dim))
            lastOutput = dim
        return res
    def __init__(self,inputdim = 768,laten_z_num = 3,
        encode_hidden_dims=[8000,3000,1000,1000],
        decode_hidden_dims=[],
        dtype = None):
        """
            decode_hidden_dims = []:解码器和编码器反对称
        """
        super(VAE, self).__init__()

        self.inputdim = inputdim
        self.deivce = None
        self.dtype = dtype
        self.decoder_output_acf = torch.sigmoid
        self.recon_loss_fc = F.mse_loss
        encode_dims = [inputdim] + encode_hidden_dims
        decode_dims = [laten_z_num] + encode_dims[::-1]

        self.encoder_mlp = VAE.CreateMLP(encode_dims)
        self.encoder_mu_logvar = torch.nn.Linear(encode_hidden_dims[-1], laten_z_num * 2)

        self.decoder_mlp = VAE.CreateMLP(decode_dims)

    def __softmax__(x):
        return torch.softmax(x,dim = 1)

    def toInference(net,x):
        mu_logvar = net.encoder(x)
        # print(mu_logvar)
        try:
            mu,logvar = mu_logvar.chunk(2,dim=1)
        except:
            mu,logvar = mu_logvar[0].chunk(2,dim=1)

        z = net.reparameterise(mu, logvar)
        z = z.exp() / z.exp().sum()

        return z

    def inference(self,x):
        mu_logvar = self.encoder(x)
        mu,logvar = mu_logvar.chunk(2,dim=0)
        z = self.reparameterise(mu, logvar)
        z = F.softmax(z)
        return z

    def encoder(self,x):

        e = F.relu(self.encoder_mlp[0](x))
        for ln in self.encoder_mlp[1:]:
            e = F.relu(ln(e))

        return self.encoder_mu_logvar(e)

    def decoder(self,rez):

        e = F.relu(self.decoder_mlp[0](rez))
        for ln in self.decoder_mlp[1:-1]:
            e = F.relu(ln(e))
        e = self.decoder_output_acf(self.decoder_mlp[-1](e))
        return e

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        # print(mu.shape,logvar.shape)
        return mu + epsilon * torch.exp(logvar / 2)

    def loss(x,recon_x, mu, logvar,recon_loss_fc = F.mse_loss):

        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar)))
        # print(recon_x.size(),x.size())
        recon_loss = recon_loss_fc(recon_x,x)
        return ( recon_loss + kl_loss ) / recon_x.size(0)


    def forward(self, x):

        mu_logvar = self.encoder(x)
        # print(mu_logvar.shape)
        mu,logvar = mu_logvar.chunk(2,dim=1)
        # print(mu)
        # print(logvar)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar
