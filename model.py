import torch
import torch.nn as nn
import torch.nn.functional as F


class KaimingConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.Parameter(self.weight)

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class KaimingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.Parameter(self.weight)
        nn.Parameter(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# NOTE: We use non affine batch norm
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class HasAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(HasAffineBatchNorm, self).__init__(dim, affine=True)

class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        self.conv1 = KaimingConv(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = HasAffineBatchNorm(64)
        self.conv2 = KaimingConv(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = HasAffineBatchNorm(64)
        self.conv3 = KaimingConv(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = HasAffineBatchNorm(128)
        self.conv4 = KaimingConv(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn4 = HasAffineBatchNorm(128)
        self.conv5 = KaimingConv(128, 256, 3, stride=1, padding=1, bias=False)
        self.bn5 = HasAffineBatchNorm(256)
        self.conv6 = KaimingConv(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn6 = HasAffineBatchNorm(256)
        self.fc1 = KaimingLinear(4*4*256, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc2 = KaimingLinear(256, 256, bias=False)
        self.bn8 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc3 = KaimingLinear(256, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn8(x)
        x = F.relu(x)
        output = self.fc3(x)
        #output = F.log_softmax(x, dim=1)
        return output

class ResNet18(nn.Module):
    channels1 = 32
    channels2 = 64
    channels3 = 128
    channels4 = 256
    def __init__(self):
        super(ResNet18, self).__init__()
        #layer1
        self.conv1 = KaimingConv(3, self.channels1, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = HasAffineBatchNorm(self.channels1)

        ##layer2
        self.conv2_1 = KaimingConv(self.channels1, self.channels1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = HasAffineBatchNorm(self.channels1)
        self.conv2_2 = KaimingConv(self.channels1, self.channels1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = HasAffineBatchNorm(self.channels1)
        self.conv2_3 = KaimingConv(self.channels1, self.channels1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_3 = HasAffineBatchNorm(self.channels1)
        self.conv2_4 = KaimingConv(self.channels1, self.channels1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_4 = HasAffineBatchNorm(self.channels1)

        ##layer3
        self.downsample3_1 = KaimingConv(self.channels1, self.channels2, kernel_size=1, stride=2, padding=0, bias=False) #residualのchannel数とfeature map sizeをそろえる
        self.dbn3_1 = HasAffineBatchNorm(self.channels2)

        self.conv3_1 = KaimingConv(self.channels1, self.channels2, kernel_size=3, stride=2, padding=1, bias=False) #ここでfeature mapサイズは半減
        self.bn3_1 = HasAffineBatchNorm(self.channels2)
        self.conv3_2 = KaimingConv(self.channels2, self.channels2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = HasAffineBatchNorm(self.channels2)
        self.conv3_3 = KaimingConv(self.channels2, self.channels2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_3 = HasAffineBatchNorm(self.channels2)
        self.conv3_4 = KaimingConv(self.channels2, self.channels2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_4 = HasAffineBatchNorm(self.channels2)

        ##layer4
        self.downsample4_1 = KaimingConv(self.channels2, self.channels3, kernel_size=1, stride=2, padding=0, bias=False) #residualのchannel数とfeature map sizeをそろえる
        self.dbn4_1 = HasAffineBatchNorm(self.channels3)

        self.conv4_1 = KaimingConv(self.channels2, self.channels3, kernel_size=3, stride=2, padding=1, bias=False) #ここでfeature mapサイズは半減
        self.bn4_1 = HasAffineBatchNorm(self.channels3)
        self.conv4_2 = KaimingConv(self.channels3, self.channels3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = HasAffineBatchNorm(self.channels3)
        self.conv4_3 = KaimingConv(self.channels3, self.channels3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_3 = HasAffineBatchNorm(self.channels3)
        self.conv4_4 = KaimingConv(self.channels3, self.channels3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_4 = HasAffineBatchNorm(self.channels3)

        ##layer5
        self.downsample5_1 = KaimingConv(self.channels3, self.channels4, kernel_size=1, stride=2, padding=0, bias=False) #residualのchannel数とfeature map sizeをそろえる
        self.dbn5_1 = HasAffineBatchNorm(self.channels4)

        self.conv5_1 = KaimingConv(self.channels3, self.channels4, kernel_size=3, stride=2, padding=1, bias=False) #ここでfeature mapサイズは半減
        self.bn5_1 = HasAffineBatchNorm(self.channels4)
        self.conv5_2 = KaimingConv(self.channels4, self.channels4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2 = HasAffineBatchNorm(self.channels4)
        self.conv5_3 = KaimingConv(self.channels4, self.channels4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_3 = HasAffineBatchNorm(self.channels4)
        self.conv5_4 = KaimingConv(self.channels4, self.channels4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_4 = HasAffineBatchNorm(self.channels4)


        ##last layer
        self.avgpool = nn.AvgPool2d(4) #feature mapを1x1サイズになるように平均化
        self.fc = KaimingLinear(self.channels4, 10, bias=True)


    def forward(self, x):

        ##layer1
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, inplace=True)

        ##layer2
        residual2_1 = out1

        out2_1 = self.conv2_1(out1)
        out2_1 = self.bn2_1(out2_1)
        out2_1 = F.relu(out2_1, inplace=True)

        out2_1 = self.conv2_2(out2_1)
        out2_1 = self.bn2_2(out2_1)
        out2_1 += residual2_1
        out2_1 = F.relu(out2_1, inplace=True)

        residual2_2 = out2_1

        out2_2 = self.conv2_3(out2_1)
        out2_2 = self.bn2_3(out2_2)
        out2_2 = F.relu(out2_2, inplace=True)

        out2_2 = self.conv2_4(out2_2)
        out2_2 = self.bn2_4(out2_2)
        out2_2 += residual2_2
        out2_2 = F.relu(out2_2, inplace=True)

        ##layer3
        residual3_1 = self.downsample3_1(out2_2)
        residual3_1 = self.dbn3_1(residual3_1)

        out3_1 = self.conv3_1(out2_2)
        out3_1 = self.bn3_1(out3_1)
        out3_1 = F.relu(out3_1, inplace=True)

        out3_1 = self.conv3_2(out3_1)
        out3_1 = self.bn3_2(out3_1)
        out3_1 += residual3_1
        out3_1 = F.relu(out3_1, inplace=True)

        residual3_2 = out3_1

        out3_2 = self.conv3_3(out3_1)
        out3_2 = self.bn3_3(out3_2)
        out3_2 = F.relu(out3_2, inplace=True)

        out3_2 = self.conv3_4(out3_2)
        out3_2 = self.bn3_4(out3_2)
        out3_2 += residual3_2
        out3_2 = F.relu(out3_2, inplace=True)

        ##layer4
        residual4_1 = self.downsample4_1(out3_2)
        residual4_1 = self.dbn4_1(residual4_1)

        out4_1 = self.conv4_1(out3_2)
        out4_1 = self.bn4_1(out4_1)
        out4_1 = F.relu(out4_1, inplace=True)

        out4_1 = self.conv4_2(out4_1)
        out4_1 = self.bn4_2(out4_1)
        out4_1 += residual4_1
        out4_1 = F.relu(out4_1, inplace=True)

        residual4_2 = out4_1

        out4_2 = self.conv4_3(out4_1)
        out4_2 = self.bn4_3(out4_2)
        out4_2 = F.relu(out4_2, inplace=True)

        out4_2 = self.conv4_4(out4_2)
        out4_2 = self.bn4_4(out4_2)
        out4_2 += residual4_2
        out4_2 = F.relu(out4_2, inplace=True)

        ##layer4
        residual5_1 = self.downsample5_1(out4_2)
        residual5_1 = self.dbn5_1(residual5_1)

        out5_1 = self.conv5_1(out4_2)
        out5_1 = self.bn5_1(out5_1)
        out5_1 = F.relu(out5_1, inplace=True)

        out5_1 = self.conv5_2(out5_1)
        out5_1 = self.bn5_2(out5_1)
        out5_1 += residual5_1
        out5_1 = F.relu(out5_1, inplace=True)

        residual5_2 = out5_1

        out5_2 = self.conv5_3(out5_1)
        out5_2 = self.bn5_3(out5_2)
        out5_2 = F.relu(out5_2, inplace=True)

        out5_2 = self.conv5_4(out5_2)
        out5_2 = self.bn5_4(out5_2)
        out5_2 += residual5_2
        out5_2 = F.relu(out5_2, inplace=True)

        ##last layer
        output = self.avgpool(out5_2)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output