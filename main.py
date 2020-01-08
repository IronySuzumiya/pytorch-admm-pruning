from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from optimizer import PruneAdam
from model import LeNet, AlexNet
from utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune, show_statistic_result
from torchvision import datasets, transforms
#from tqdm import tqdm
import os
import time

def pre_train(args, model, device, train_loader, test_loader, optimizer):
    pre_train_start = time.time()
    for epoch in range(args.num_pre_epochs):
        print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        epoch_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.step()
        epoch_end = time.time()
        print("pre-train epoch time cost: {}".format(epoch_end - epoch_start))
        test(args, model, device, test_loader)
    pre_train_end = time.time()
    print("pre-train total time cost: {}".format(pre_train_end - pre_train_start))


def train(args, model, device, train_loader, test_loader, optimizer):
    train_start = time.time()
    Z, U = initialize_Z_and_U(model, device)
    for epoch in range(args.num_epochs):
        print('Epoch: {}'.format(epoch + 1))
        model.train()
        epoch_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            loss.backward()
            optimizer.step()
        epoch_end = time.time()
        print("train epoch time cost: {}".format(epoch_end - epoch_start))
        admm_step_start = time.time()
        X = update_X(model, device)
        Z = update_Z_l1(X, U, args) if args.l1 else update_Z(X, U, args, device)
        U = update_U(U, X, Z)
        admm_step_end = time.time()
        print("admm step time cost: {}".format(admm_step_end - admm_step_start))
        print_convergence(model, X, Z)
        test(args, model, device, test_loader)
    train_end = time.time()
    print("train total time cost: {}".format(train_end - train_start))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    retrain_start = time.time()
    for epoch in range(args.num_re_epochs):
        print('Re epoch: {}'.format(epoch + 1))
        model.train()
        epoch_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.prune_step(mask)
        epoch_end = time.time()
        print("retrain epoch time cost: {}".format(epoch_end - epoch_start))
        test(args, model, device, test_loader)
    retrain_end = time.time()
    print("retrain total time cost: {}".format(retrain_end - retrain_start))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "cifar10"],
                        metavar='D', help='training dataset (mnist or cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--percent', type=list, default=[0.8, 0.92, 0.991, 0.93],
                        metavar='P', help='pruning percentage (default: 0.8)')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L',
                        help='l2 norm weight (default: 5e-4)')
    parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--l1', default=False, action='store_true',
                        help='prune weights with l1 regularization instead of cardinality')
    parser.add_argument('--l2', default=False, action='store_true',
                        help='apply l2 regularization')
    parser.add_argument('--num_pre_epochs', type=int, default=3, metavar='P',
                        help='number of epochs to pretrain (default: 3)')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R',
                        help='number of epochs to retrain (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E',
                        help='adam epsilon (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--struct', action='store_true', default=False,
                        help='Enabling struct Pruning')
    parser.add_argument('--test', action='store_true', default=False,
                        help='For Testing the current Model')
    parser.add_argument('--stat', action='store_true', default=False,
                        help='For showing the statistic result of the current Model')
    parser.add_argument('--ou-h', type=int, default=2, metavar='N',
                        help='ReRAM OU height (default: 2)')
    parser.add_argument('--ou-w', type=int, default=2, metavar='N',
                        help='ReRAM OU width (default: 2)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    else:
        args.percent = [0.8, 0.92, 0.93, 0.94, 0.95, 0.99, 0.99, 0.93]
        args.num_pre_epochs = 5
        args.num_epochs = 20
        args.num_re_epochs = 5
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.batch_size, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.test_batch_size, **kwargs)

    model = LeNet().to(device) if args.dataset == "mnist" else AlexNet().to(device)
    optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

    struct_tag = "_struct{}x{}".format(args.ou_h, args.ou_w) if args.struct else ""

    model_file = "mnist_cnn{}.pt".format(struct_tag) if args.dataset == "mnist" \
            else 'cifar10_cnn{}.pt'.format(struct_tag)

    if args.stat or args.test:
        print("=> loading model '{}'".format(model_file))

        if os.path.isfile(model_file):
            model.load_state_dict(torch.load(model_file))
            print("=> loaded model '{}'".format(model_file))
            if args.test:
                test(args, model, device, test_loader)
            if args.stat:
                show_statistic_result(args, model)
        else:
            print("=> loading model failed '{}'".format(model_file))

    else:
        checkpoint_file = 'checkpoint{}.pth.tar'.format("_mnist" if args.dataset == "mnist" else "_cifar10")
        
        if not os.path.isfile(checkpoint_file):
            pre_train(args, model, device, train_loader, test_loader, optimizer)
            torch.save({
                'epoch': args.num_pre_epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_file)
        else:
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(checkpoint_file))

        train(args, model, device, train_loader, test_loader, optimizer)
        mask = apply_l1_prune(model, device, args) if args.l1 else apply_prune(model, device, args)
        print_prune(model)
        test(args, model, device, test_loader)
        retrain(args, model, mask, device, train_loader, test_loader, optimizer)

        if args.save_model:
            torch.save(model.state_dict(), model_file)

if __name__ == "__main__":
    main()