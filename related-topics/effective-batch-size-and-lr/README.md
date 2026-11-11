# 有效 batch size 与学习率

随着节点数量增加，有效 batch size（用于模型更新的样本数量）也会增加：

```
effective_batch_size = batch_size * world_size
```

你可能知道，增大 batch size 通常会降低随机梯度估计的方差，从而使梯度更平滑（不是改变数据集本身的方差）。这会直接影响模型学习和变化的动态。

如果从单卡训练迁移到多节点训练时，希望**精确匹配单卡训练的动态**，本章就是为你准备的。

> 译注：上述公式假设纯数据并行、每个 rank 的 batch size 相同且没有梯度累积。一般应使用 `每个数据并行 rank 的 batch size × DP组大小 × 梯度累积步数`。TP 组里的各 rank 处理相同输入，不能重复计入。下面的学习率缩放是经验规则，不保证训练轨迹或效果与单卡精确一致。

## 缩放规则

如果希望训练动态完全一致，还必须缩放学习率。不过这取决于使用的优化器。精确规则尚未被完全理解，可以参考以下论文：

- [Exploring Learning Rate Scaling Rules for Distributed ML Training on Transient Resources](https://anakli.inf.ethz.ch/papers/learning_rate_distribml22.pdf)

撰写本文时，人们最常用的学习率缩放规则是：

### 线性缩放规则

```python
lr = args.lr * dist.get_world_size()
```

它最早出现在上面提到的大批量 SGD 论文中。不过它并不能完全产生相同的训练动态，论文实际使用的是 **world size 的某个系数**。

注意：**对 SGD 以外的优化器使用这条规则时要小心。**

参考资料：

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677)

### 平方根缩放规则

```python
lr = args.lr * numpy.sqrt(dist.get_world_size())
```

这条规则建议用于 Adam 优化器。扩大 batch 数量时，它保持梯度方差的平方根。

参考资料：

- [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/pdf/1404.5997)
- [Large-Batch Training for LSTM and Beyond](https://arxiv.org/pdf/1901.08256)
