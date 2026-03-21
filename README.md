# 机器学习遗忘 (Machine Unlearning)

## 简介
本仓库专注于机器学习单个样本遗忘算法的实现，使用PyTorch。

## 项目架构
- **训练管道**: `train_main.py` 使用 `src.dataset.get_dataset` -> `src.raw_dataset` 加载数据，通过 `getattr(models, args.model)` 构造模型，使用 `src.utils.save_model` 训练并保存。
- **遗忘管道**: 单个样本遗忘通过 `scripts/forget_one.py` 和 `scripts/forget_one_blend.py` 实现，使用 `unlearn_strategies.unlearn.blindspot_unlearner` 进行遗忘。
- **模型注册**: `model/models.py` 暴露工厂函数 (`ResNet18`, `SimpleCNN`, `MLP`)，通过 `getattr(models, name)` 使用。
- **策略注册**: 简化的 `unlearn_strategies/unlearn.py` 包含单个样本遗忘的核心函数。

## 开始使用

### 准备工作

在执行项目代码之前，请根据 `requirements.txt` 文件准备Python环境。我们使用 `python 3.9.12` 和 `torch 2.0.0` 设置环境。

```bash
pip install -r requirements.txt
```

### 如何运行

**1. 模型训练**

```bash
python train_main.py -dataset Cifar10 -gpu True -epochs 30 -batch_size 128
```

参数说明：
- `-dataset`: 数据集名称 (MNist, FMNist, Cifar10, Cifar100, TinyImagenet)
- `-gpu`: 是否使用GPU进行训练 (True/False)
- `-epochs`: 训练轮数
- `-batch_size`: 每批处理的样本数量
- `-model`: 模型架构名称 (默认: ResNet18，可选: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SimpleCNN, MLP, LRTorchNet)

**2. 单个样本遗忘**

```bash
python scripts/forget_one.py --dataset Cifar10 --model ResNet18 --model_path checkpoint/... --forget_idx 0
python scripts/forget_one_blend.py --dataset Cifar10 --model ResNet18 --model_path checkpoint/... --forget_idx 0 --mix_mode block --block_size 4
```

参数说明：
- `--dataset`: 数据集名称 (MNist, FMNist, Cifar10, Cifar100, TinyImagenet)
- `--model`: 模型架构名称 (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, SimpleCNN, MLP, LRTorchNet)
- `--model_path`: 预训练模型检查点文件路径，需要手动去checkpoint复制
- `--forget_idx`: 要遗忘的训练样本索引
- `--mix_mode`: 混合模式 (none: 无混合, direct: 直接混合, block: 块状混合)
- `--block_size`: 块大小 (仅在block混合模式下使用)

### 项目结构
- `train_main.py`: 训练入口点
- `scripts/forget_one.py`: 单个样本遗忘脚本（使用blindspot unlearner）
- `scripts/forget_one_blend.py`: 增强的单个样本遗忘脚本（结合protect set和mixing）
- `scripts/eval_forget.py`: 遗忘效果评估脚本
- `scripts/plot_unlearn_csv.py`: 从CSV日志生成可视化图表
- `unlearn_strategies/unlearn.py`: 遗忘核心函数（blindspot_unlearner等）
- `model/models.py` 和 `model/resnet.py`: 模型工厂
- `src/dataset.py` 和 `src/raw_dataset.py`: 数据加载
- `src/utils.py`: 辅助函数 (种子/设备/保存)
- `src/protect_selection.py`: protect set选择逻辑（用于单个样本遗忘）
- `checkpoint/`: 保存的模型检查点
- `data/`: 数据集
- `logs/`: 实验日志（CSV格式，记录遗忘效果指标）
- `plots/`: 可视化图表结果

## 单个样本遗忘 (Sample-Level Unlearning)

项目支持遗忘单个训练样本，使用两种不同的方法：

### 方法1: 基础遗忘 (`forget_one.py`)
使用blindspot unlearning算法进行单个样本遗忘。

**运行命令：**
```bash
python scripts/forget_one.py --dataset Cifar10 --model ResNet18 --model_path checkpoint/Cifar10/ResNet18/baseline_...pt --forget_idx 0
```

**输出文件：**
- 遗忘后的模型保存为：`checkpoint/forgot_idx_0.pt`

### 方法2: 增强遗忘 (`forget_one_blend.py`)
结合protect set（保留锚点）和mixing注入，提供更强的遗忘保证。

**运行命令：**
```bash
python scripts/forget_one_blend.py --dataset Cifar10 --model ResNet18 --model_path checkpoint/Cifar10/ResNet18/baseline_...pt --forget_idx 0 --mix_mode block --block_size 4
```

**输出文件举例：**
- 遗忘后的模型保存为：`checkpoint/Cifar10/ResNet18/forget_idx0_mixblock_lamNA_blk4_aw10.0_rw0.0001_ep10_bs64_exp1_20260129_002728.pt`
- CSV日志文件保存为：`logs/unlearn_Cifar10_SimpleCNN_idx0_mixblock_lamNA_blk4_aw10.0_rw0.0001_ep10_bs64_exp1_20260129_002728.csv`

### 完整使用流程

1. **训练基础模型**（如果还没有的话）：
   ```bash
   python train_main.py -dataset Cifar10 -gpu True -epochs 30 -batch_size 128
   ```
   模型将保存到 `checkpoint/Cifar10/ResNet18/` 目录下。

2. **执行遗忘**：
   选择其中一种遗忘方法运行。

3. **评估遗忘效果**：
   ```bash
   python scripts/eval_forget.py --dataset Cifar10 --model_path checkpoint/forgot_idx_0.pt --forget_idx 0
   ```
   这将输出遗忘前后的准确率对比和MIA分数。

### 评估遗忘效果
```bash
python scripts/eval_forget.py --dataset Cifar10 --model_path checkpoint/forgot_idx_0.pt --forget_idx 0
```

评估脚本会比较原始模型和遗忘后模型的性能，输出：
- 保留集准确率（应该保持较高）
- 遗忘样本准确率（应该显著降低）
- MIA分数（成员推理攻击成功率，应该降低）

## 评估指标详解

遗忘效果通过三个关键指标进行量化评估：

### 1. 保留集准确率 (Retain Accuracy)
- **定义**: 模型在保留数据集（未被遗忘的数据）上的分类准确率
- **理想表现**: 应该保持与原始模型相近的高准确率
- **意义**: 确保遗忘过程不会过度影响模型在其他数据上的性能

### 2. 遗忘准确率 (Forget Accuracy) 
- **定义**: 模型在要遗忘的特定样本上的分类准确率
- **理想表现**: 应该显著降低（接近随机猜测水平）
- **意义**: 验证模型确实"忘记"了指定的样本，无法再正确分类该样本

### 3. MIA分数 (Membership Inference Attack Score)
- **定义**: 使用成员推理攻击评估遗忘效果的指标
- **计算方法**: 
  - 收集模型对保留集、遗忘集和测试集的预测概率
  - 计算每个样本预测概率的熵值
  - 使用逻辑回归分类器训练，区分"成员"（训练数据）和"非成员"（测试数据）
  - 在遗忘样本上测试分类器的预测准确率
- **理想表现**: 应该接近50%（随机猜测水平）
- **意义**: 如果MIA分数显著高于50%，说明攻击者仍能通过模型行为推断出样本是否曾用于训练，遗忘不彻底

**MIA分数解释**:
- 50%: 完全随机，无法区分成员和非成员（理想的遗忘效果）
- >70%: 攻击者能较好地区分，遗忘效果不佳
- <50%: 模型对遗忘样本的预测过于保守（可能过度遗忘）

## 可视化和日志

### CSV日志文件
遗忘过程会生成详细的CSV日志文件，记录训练过程中的各项指标：

**文件位置：** `logs/` 目录
**命名格式：** `unlearn_{dataset}_{model}_idx{forget_idx}_{mix_mode}_lam{lambda}_blk{block_size}_aw{ascent_w}_rw{retain_w}_ep{epochs}_bs{batch_size}_{timestamp}.csv`

**CSV内容包括：**
- `phase`: 阶段（baseline: 基准线, train: 训练过程）
- `epoch`: 训练轮数
- `test_acc`: 测试集准确率
- `forget_pred`: 遗忘样本的预测结果
- `p_true`: 遗忘样本真实类别的预测概率
- `p_max`: 预测概率最高的类别概率
- `margin`: 预测概率最高的两个类别之间的差距
- `CE_forget`: 遗忘损失
- `CE_retain`: 保留损失

### 可视化图表
使用CSV日志生成遗忘效果的可视化图表：

**运行命令：**
```bash
python scripts/plot_unlearn_csv.py --log_dir ./logs --out_dir ./plots
```

**输出文件：**
- 测试准确率随训练轮数的变化图：`plots/unlearn_test_acc.png`
- 遗忘样本真实类别预测概率的变化图：`plots/unlearn_p_true.png`

**图表位置：** `plots/` 目录

### 如何查找文件
1. **模型文件**：在 `checkpoint/` 目录下，按数据集和模型类型组织
2. **CSV日志**：在 `logs/` 目录下，按时间戳排序找到最新的日志文件
3. **图表文件**：在 `plots/` 目录下，PNG格式的图像文件

**提示：** 文件名中的时间戳格式为 `YYYYMMDD_HHMMSS`，可以帮助您找到最新的实验结果。
