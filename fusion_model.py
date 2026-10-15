import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self, num_classes, text_dim):
        super(FusionModel, self).__init__()

        # Image branch: 512-d → 256-d
        self.img_fc = nn.Linear(512, 256)
        self.img_bn = nn.BatchNorm1d(256)

        # Text branch: text_dim → 256-d
        self.text_fc = nn.Linear(text_dim, 256)
        self.text_bn = nn.BatchNorm1d(256)

        # Fusion classifier
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, img_feat, text_feat):
        img_out = F.relu(self.img_bn(self.img_fc(img_feat)))
        text_out = F.relu(self.text_bn(self.text_fc(text_feat)))

        x = torch.cat((img_out, text_out), dim=1)  # 512-d

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x