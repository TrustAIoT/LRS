import argparse
import copy
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import models
from utils.ft import llr, tulip, cure

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='lra1')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lam1', type=float, default=5.0)
parser.add_argument('--step_size1', type=float, default=0.01)
parser.add_argument('--lam2', type=float, default=5)
parser.add_argument('--step_size2', type=float, default=1.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save-dir', type=str, default='finetuned')
args = parser.parse_args()

logging.basicConfig(filename='logs.log', level=logging.INFO)
logging.info("Finetuning")
logging.info(args)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return test_loss, acc

def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    args.constraint = "linf"
    args.niters = 10
    args.epsilon = 8 / 255.
    args.step_size = 1 / 255.
    
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    #model = models.__dict__['resnet50'](pretrained=True)
    #model.load_state_dict(torch.load('models/resnet/resnet50.pt', map_location='cpu'))
    model = models.__dict__['densenet'](
                num_classes=10,
                depth=100,
                growthRate=12,
                compressionRate=2,
                dropRate=0,
            )
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('models/densenet-bc-L100-k12/model_best.pth.tar', map_location=device)['state_dict'])
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(args.epochs):
        model.train()
        for i, (img, label) in enumerate(trainloader):
            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad()
            loss_fn = nn.CrossEntropyLoss()

            #output, loss = llr(model, loss_fn, img, label, optimizer)
            if args.method == 'lra1':
                output, loss = tulip(model, loss_fn, img, label, args.step_size1, args.lam1)
            if args.method == 'lra2':
                output, loss = cure(model, loss_fn, img, label, args.step_size2, args.lam2)
            if args.method == 'lra12':
                output, loss = tulip(model, loss_fn, img, label, args.step_size1, args.lam1) + cure(model, loss_fn, img, label, args.step_size2, args.lam2)
                
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                acc = 100*(output.argmax(1) == label).sum() / len(img)
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.2f}'.format(
                    epoch, i * len(img), len(trainloader.dataset),
                        100. * i / len(trainloader), loss.item(), acc))

        loss_cln_eval, acc_eval = test(model, testloader, device)
        logging.info('CURRENT EVAL Loss: {:.6f}\tAcc:{:.2f}'.format(loss_cln_eval, acc_eval))

        torch.save({"state_dict": model.state_dict(),
                    "opt_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                    os.path.join(args.save_dir, args.method, 'ep_{}.pt'.format(epoch)))
    
if __name__ == '__main__':
    main()

