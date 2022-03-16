import torch
import torch.nn as nn

class CombinedNet(nn.Module):
    def __init__(self, deepset, image_cnn, classifier_layers):
        super(CombinedNet, self).__init__()
        self.deepset = deepset
        self.image_cnn = image_cnn
        self.bn_layers = nn.ModuleList([])
        self.class_layers = nn.ModuleList([])
        self.activ = nn.ReLU()
        for hidden_i in range(1, len(classifier_layers)):
            self.class_layers.append(nn.Linear(classifier_layers[hidden_i - 1], classifier_layers[hidden_i]))
            self.bn_layers.append(nn.BatchNorm1d(classifier_layers[hidden_i]))

    def forward(self, tracks, cell_image):
        x1 = self.deepset(tracks)
        x2 = self.image_cnn(cell_image)
        #x2_flat = torch.flatten(x2)
        x = torch.cat((x2, x1), dim=1)
        for i in range(len(self.class_layers)):
            x = self.class_layers[i](x)
            if x.shape[0] > 1:
                x = self.bn_layers[i](x)
            if i < len(self.class_layers) - 1:
                x = self.activ(x)

        return x