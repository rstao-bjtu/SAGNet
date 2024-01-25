from .clip import clip 
from PIL import Image
import torch.nn as nn
from torch.autograd import Function
# from .resnet import ResNet, Bottleneck, BasicBlock


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class DANNModel(nn.Module):
    def __init__(self, input_size=2048):
        super(DANNModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 1)
        )
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        # Domain prediction
        features_reverse = grad_reverse(features)
        domain_output = self.domain_classifier(features_reverse)
        return domain_output

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.neg() * x.new_ones(x.size())
        return grad_input, None

def grad_reverse(x):
    return GradReverse.apply(x)

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # self.resnet = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.Domain_classifier = DANNModel(CHANNELS[name])
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward_train(self, x, label, return_feature=False):
        features = self.model.encode_image(x)
        y = features[(label == 0).nonzero().squeeze()]
        domain_output = self.Domain_classifier(y)
        if return_feature:
            return features
        return self.fc(features), domain_output

    def forward_test(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

