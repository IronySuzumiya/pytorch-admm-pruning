# pytorch-admm-prunning
It is a pytorch implementation of DNN weight prunning with ADMM described in [**A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers**](https://arxiv.org/abs/1804.03294).

## _Train and test_
- You can simply run code by
```
$ python main.py --save-model
```

- In the paper, authors use **l2-norm regularization** so you can easily add by
```
$ python main.py --l2 --save-model
```

- Beyond this paper, if you don't want to use _predefined prunning ratio_, admm with **l1 norm regularization** can give a great solution and can be simply tested by
```
$ python main.py --l1
```

- There are two dataset you can test in this code: **[mnist, cifar10]**. Default setting is mnist, you can change dataset by
```
$ python main.py --l2 --save-model --dataset cifar10
```

- To enable ReRAM-adapted structural ADMM pruning, run code by
```
$ python main.py --l2 --struct --save-model
```

- You can change ReRAM OU size by
```
$ python main.py --l2 --struct --ou-h=4 --ou-w=4 --save-model
```

- You can check the statistic result by
```
$ python main.py --l2 --struct --stat
```

- You can test the current trained model by
```
$ python main.py --l2 --struct --test
```

## _Models_
- In this code, there are two models: **[LeNet, AlexNet]**. I use LeNet for mnist, AlexNet for cifar10 by default.

## _Optimizer_
- To prevent prunned weights from updated by optimizer, I modified Adam (named PruneAdam).

## _References_
For this repository, I refer to _[KaiqiZhang's tensorflow implementation](https://github.com/KaiqiZhang/admm-pruning)_.
