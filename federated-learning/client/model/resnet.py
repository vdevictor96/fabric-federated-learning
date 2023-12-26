
#     import torch
# import torch.nn as nn
# import torch.optim as optim

# import torch.nn.functional as F
# from model.simple import SimpleNet
# from torch.autograd import Variable

    
    # class ResNet(SimpleNet):
    # def __init__(self, block, num_blocks, class_num=10, name=None, created_time=None):
    #     super(ResNet, self).__init__()
    #     self.in_planes = 32

    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    #     self.bn1 = nn.BatchNorm2d(32)
    #     self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
    #     self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
    #     self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
    #     self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
    #     self.linear = nn.Linear(256*block.expansion, class_num)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out
    