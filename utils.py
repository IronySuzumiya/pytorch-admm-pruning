import torch
import torch.nn.functional as F
import numpy as np
import collections

def regularized_nll_loss(args, model, output, target):
    index = 0
    loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                loss += args.alpha * param.norm()
                index += 1
    return loss


def admm_loss(args, device, model, Z, U, output, target):
    idx = 0
    loss = F.nll_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            u = U[idx].to(device)
            z = Z[idx].to(device)
            loss += args.rho / 2 * (param - z + u).norm()
            if args.l2:
                loss += args.alpha * param.norm()
            idx += 1
    return loss


def initialize_Z_and_U(model):
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
    return Z, U


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            X += (param.detach().cpu().clone(),)
    return X


def scale(input, shape, n1, n2):
    output = torch.zeros((shape[0], shape[1]), dtype=torch.bool)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            output[i * n1 : (i + 1) * n1, j * n2 : (j + 1) * n2] = input[i, j]
    return output


def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB


def update_Z(X, U, args):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u
        if args.structured:
            rram = z.view(z.shape[0], -1).T
            tmp = torch.zeros(((rram.shape[0] - 1) // args.n1 + 1, (rram.shape[1] - 1) // args.n2 + 1))
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i, j] = rram[i * args.n1 : (i + 1) * args.n1, j * args.n2 : (j + 1) * args.n2].norm()
            pcen = np.percentile(tmp, 100*args.percent[idx])
            upon_threshold = tmp >= pcen
            res1 = rram.shape[0] % args.n1
            res2 = rram.shape[1] % args.n2
            for i in range(args.n1):
                for j in range(args.n2):
                    if i < res1 or res1 == 0:
                        rram.data[i::args.n1, j::args.n2] *= upon_threshold if j < res2 or res2 == 0 else upon_threshold[:, :-1]
                    else:
                        rram.data[i::args.n1, j::args.n2] *= upon_threshold[:-1, :] if j < res2 or res2 == 0 else upon_threshold[:-1, :-1]
            #under_threshold = scale(tmp < pcen, rram.shape, args.n1, args.n2)
            #rram.data[under_threshold] = 0
        else:
            pcen = np.percentile(abs(z), 100*args.percent[idx])
            under_threshold = abs(z) < pcen
            z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1
    return new_Z


def update_Z_l1(X, U, args):
    new_Z = ()
    delta = args.alpha / args.rho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def prune_weight(args, param, device, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    if args.structured:
        weight = param.detach().to(device).clone()
        mask = torch.zeros_like(weight, dtype=torch.bool).to(device)
        rram = weight.view(weight.shape[0], -1).T
        rram_mask = mask.view(mask.shape[0], -1).T
        tmp = torch.zeros(((rram.shape[0] - 1) // args.n1 + 1, (rram.shape[1] - 1) // args.n2 + 1))
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i, j] = rram[i * args.n1 : (i + 1) * args.n1, j * args.n2 : (j + 1) * args.n2].norm()
        pcen = np.percentile(tmp, 100*percent)
        upon_threshold = tmp >= pcen
        res1 = rram.shape[0] % args.n1
        res2 = rram.shape[1] % args.n2
        for i in range(args.n1):
            for j in range(args.n2):
                if i < res1 or res1 == 0:
                    rram_mask.data[i::args.n1, j::args.n2] = upon_threshold if j < res2 or res2 == 0 else upon_threshold[:, :-1]
                else:
                    rram_mask.data[i::args.n1, j::args.n2] = upon_threshold[:-1, :] if j < res2 or res2 == 0 else upon_threshold[:-1, :-1]
                rram.data[i::args.n1, j::args.n2] *= rram_mask.data[i::args.n1, j::args.n2]
        #under_threshold = scale(tmp < pcen, rram_proj.shape, args.n1, args.n2)
        #rram_proj.data[under_threshold] = 0
    else:
        weight = param.detach().cpu().clone()
        pcen = np.percentile(abs(weight), 100*percent)
        under_threshold = abs(weight) < pcen
        weight.data[under_threshold] = 0
        mask = (abs(weight) >= pcen).to(device)

    return mask


def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask


def apply_prune(model, device, args):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_weight(args, param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def apply_l1_prune(model, device, args):
    delta = args.alpha / args.rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))

def show_statistic_result(args, model):
    n_non_zero = collections.OrderedDict()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            rram_proj = param.detach().cpu().clone().view(param.shape[0], -1).T
            for i in range((rram_proj.shape[0] - 1) // args.n1 + 1):
                for j in range((rram_proj.shape[1] - 1) // args.n2 + 1):
                    ou = rram_proj[i * args.n1 : (i + 1) * args.n1, j * args.n2 : (j + 1) * args.n2]
                    n = ou.nonzero().shape[0]
                    if n in n_non_zero:
                        n_non_zero[n] += 1
                    else:
                        n_non_zero[n] = 1
    print("n_non_zero: {}".format(n_non_zero))
