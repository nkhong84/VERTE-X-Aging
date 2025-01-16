from math import pi, cos

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def load_optimizer(model, args):
    if not args.bias_decay:
        weight_params = []
        bias_params = []
        for n, p in model.named_parameters():
            if 'bias' in n:
                bias_params.append(p)
            else:
                weight_params.append(p)
        parameters = [{'params' : bias_params, 'weight_decay' : 0},
                      {'params' : weight_params}]
    else:
        parameters = model.parameters()

    if args.optim.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
  
    return optimizer
def load_loss_function(args,lambda_1, lambda_2, start_age, end_age):
    criterion = MeanVariance_CrossELoss(lambda_1, lambda_2, start_age, end_age)



class MeanVariance_CrossELoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.loss1 = MeanVarianceLoss(lambda_1, lambda_2, start_age, end_age)
        self.loss2 = nn.CrossEntropyLoss()
        self.start_age = start_age
        
    def forward(self, pred, target):
        mean_loss, variance_loss = self.loss1(pred, target)
        softmax_loss = self.loss2(pred, target-self.start_age) # list start from 0
        
        return mean_loss + variance_loss + softmax_loss,mean_loss, variance_loss, softmax_loss

        
class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        
        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss
                