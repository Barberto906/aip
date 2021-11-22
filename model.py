from torch import nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, resnet_type=50, output_size=1024, mlp_on_top=True):
        super(Net, self).__init__()
        self.name = ""
        if resnet_type == 50:
            self.backbone = models.resnet50(pretrained=True)
            self.name = "resnet50"
        elif resnet_type == 18:
            self.backbone = models.resnet18(pretrained=True)
            self.name = "resnet18"
        elif resnet_type == 34:
            self.backbone = models.resnet34(pretrained=True)
            self.name = "resnet34"

        self.mlp_on_top = mlp_on_top

        if mlp_on_top:
            self.name = self.name + "_mlp_on_top"
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2048)
            self.mlp = nn.Sequential(nn.Linear(2048, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, output_size))

        else:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, output_size)

    def print_net(self):
        # for name, param in self.backbone.named_parameters():
        #     print("{} -> {}".format(name, param.shape))

        for layer in self.backbone.layer4[-1:]:
            print(layer.stride)

    def forward(self, input):
        if self.mlp_on_top:
            out = self.backbone(input)
            out = self.mlp(out)
        else:
            out = self.backbone(input)
        return out
