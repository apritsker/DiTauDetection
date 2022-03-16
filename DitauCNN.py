import torch
from torch import nn

class View(nn.Module):
    def __init__(self, shape):
            super(View, self).__init__()
            self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PrintShapeLayer(nn.Module):
    def __init__(self):
        super(PrintShapeLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class CellCNN(nn.Module):
    def __init__(self, batch_size, image_size, classify=True):
        super(CellCNN, self).__init__()
        self.classify = classify
        self.image_size = image_size
        self.image_channels = image_size[0]

        def block(in_feat, out_feat, conv_kernel, pool_kernel, normalize=False):
            layers = [nn.DataParallel(nn.Conv2d(in_feat, in_feat, conv_kernel, padding=1))]
            # if normalize:
                # layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.DataParallel(nn.Conv2d(in_feat, out_feat, conv_kernel, padding=1)))
            layers.append(nn.DataParallel(nn.MaxPool2d(pool_kernel)))
            return layers

        self.model = nn.Sequential(
            # PrintShapeLayer(),
            *block(self.image_channels, self.image_channels * 2, 3, 2, normalize=False),        # reduces W,H by factor of 2
            # PrintShapeLayer(),
            *block(self.image_channels * 2, self.image_channels * 4, 3, 2, normalize=False),    # reduces W,H by factor of 2
            # PrintShapeLayer(),
            *block(self.image_channels * 4, self.image_channels * 8, 3, 2, normalize=False),    # reduces W,H by factor of 2
            # PrintShapeLayer(),

            # View((batch_size, 4 * 4 * 16)),
            nn.DataParallel(nn.Flatten()),

            
        )

        self.linear_size = int((image_size[1] / 8) * (image_size[2] / 8) * (self.image_channels * 8))

        self.classifier = nn.Sequential(
            # PrintShapeLayer(),
            nn.DataParallel(nn.Linear(self.linear_size, self.linear_size)),
            # FlexLinear(),
            # PrintShapeLayer(),
            nn.DataParallel(nn.ReLU()),
            nn.DataParallel(nn.Linear(self.linear_size, 2)),
            nn.DataParallel(nn.Softmax(dim=1))
        )

    def forward(self, x):
        x = self.model(x)
        if(self.classify):
            x = self.classifier(x)
        return x

