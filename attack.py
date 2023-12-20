import argparse
import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from torchattacks.attacks.pgd import PGD
from torchattacks.attacks.mifgsm import MIFGSM
from torchattacks.attacks.difgsm import DIFGSM
from torchattacks.attacks.tifgsm import TIFGSM
from torchattacks.attacks.sinifgsm import SINIFGSM
import models
from utils.utils import Normalize, set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='PGD', choices=['PGD', 'MIFGSM', 'DIFGSM'])
parser.add_argument('--epsilon', type=float, default=4/255)
parser.add_argument('--step_size', type=float, default=1/255)
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--prob', type=float, default=0.5)
parser.add_argument('--clop_layer', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("./results.log"),
        logging.StreamHandler(),
    ],
)
logging.info(args)

def main():
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # modelss = []
    # if args.method == 'ori':
    #     model.load_state_dict(torch.load('models/densenet-bc-L100-k12/model_best.pth.tar', map_location=device)['state_dict'])
    # if args.method == 'lra1':
    #     for i in range(10):
    #         model.load_state_dict(torch.load('finetuned/lra1/ep_9.pt', map_location=device)['state_dict'])
    #         model = nn.Sequential(
    #             Normalize(),
    #             model
    #             )
    #         model.eval()
    #         model.to(device)
    #         modelss.append(model)
    # if args.method == 'lra2':
    #     model.load_state_dict(torch.load('finetuned/lra2/ep_9.pt', map_location=device)['state_dict'])
    # if args.method == 'lra12':
    #     model.load_state_dict(torch.load('finetuned/lra12/ep_9.pt', map_location=device)['state_dict'])

    resnet18 = models.__dict__['resnet18'](pretrained=False, prob=args.prob, clop_layer=args.clop_layer)
    state_dict = torch.load('models/resnet/resnet18.pt', map_location='cpu')
    resnet18.load_state_dict(state_dict)
    resnet18.to(device)
    resnet18.eval()
    resnet18 = nn.Sequential(
        Normalize(), 
        resnet18)
        
    # vgg19_bn = models.__dict__['vgg19_bn'](num_classes=10)
    # vgg19_bn.features = nn.DataParallel(vgg19_bn.features)
    # vgg19_bn.load_state_dict(torch.load('models/vgg19_bn/model_best.pth.tar', map_location=(device))['state_dict'])
    # vgg19_bn.to(device)
    # vgg19_bn.eval()
    # vgg19_bn = nn.Sequential(
    #     Normalize(), 
    #     vgg19_bn)

    if args.method == 'PGD':
        attack = PGD(resnet18, eps=args.epsilon, alpha=args.step_size, steps=args.niters)
    elif args.method == 'MIFGSM':
        attack = MIFGSM(resnet18, eps=args.epsilon, alpha=args.step_size, steps=args.niters)
    elif args.method == 'DIFGSM':
        attack = DIFGSM(resnet18, eps=args.epsilon, alpha=args.step_size, steps=args.niters)

    # ATTACK
    label_ls = []
    for ind, (ori_img, label) in enumerate(testloader):
        label_ls.append(label)
        ori_img, label = ori_img.to(device), label.to(device)
        adv_images = attack(ori_img, label)
        adv_images = torch.round(adv_images.data*255).cpu().numpy().astype(np.uint8())
        np.save(os.path.join(args.save_dir, 'batch_{}.npy'.format(ind)), adv_images)
        print(' batch_{}.npy saved'.format(ind))
    label_ls = torch.cat(label_ls)
    np.save(os.path.join(args.save_dir, 'labels.npy'), label_ls.numpy())
    print('images saved')


if __name__ == '__main__':
    main()
