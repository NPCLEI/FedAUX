
from torch import nn
from torch import flatten
from torchvision.models.resnet import ResNet, BasicBlock
from ..nn import NPCModule

class ResNet8(NPCModule):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()

        encoder_layers = []
        for name, module in ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                encoder_layers.append(module)
        # encoder
        self.encoder = nn.Sequential(*encoder_layers)
        # classifier
        self.classification_layer = nn.Linear(512, num_classes, bias=True)



    def extract_features(self, x):
        return flatten(self.encoder(x), start_dim=1)


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out

    def forward(self,img):
        feature = self.extract_features(img)
        feature = flatten(feature,1)
        return self.classification_layer(feature)

if __name__ == "__main__":
    import torch
    from torchvision.datasets import CIFAR10

    train = CIFAR10(r'E:\Dataset\ImgCls\CIFAR10')
    test  = CIFAR10(r'E:\Dataset\ImgCls\CIFAR10',False)

    net = ResNet8()
    net.Train(train,test)