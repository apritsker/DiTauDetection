import torch
from torch import nn

from DitauCNN import CellCNN
from DitauDeepSet import DeepSet

class DitauCombinedNet(nn.Module):
    def __init__(self, batch_size, in_features, feats, classifier_layers):
        super(DitauCombinedNet, self).__init__()

        self.cnn = CellCNN(batch_size, classify=False)
        self.deepset = DeepSet(in_features, feats, [128, 32, 8, 2], classify=False)

        self.classifier = nn.Sequential(
            # PrintShapeLayer(),
            nn.DataParallel(nn.Linear(classifier_layers[0], classifier_layers[1])),
            # FlexLinear(),
            # PrintShapeLayer(),
            nn.DataParallel(nn.ReLU()),
            nn.DataParallel(nn.Linear(classifier_layers[1], 2)),
            nn.DataParallel(nn.Softmax(dim=1))
        )


    def forward(self, tracks, cell):
        tracks_features = self.deepset(tracks)
        cells_features = self.cnn(cell)

        features_combined = torch.cat((tracks_features, cells_features), dim=1)
        
        return self.classifier(features_combined)