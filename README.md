
# CAHA-Net

本仓库包含脑肿瘤 MRI 图像分类相关的训练/评估脚本与模型实现。

说明：本仓库当前**包含** `data/` 数据集（体积较大）；训练日志、模型权重、可视化输出等仍在 `.gitignore` 中默认忽略。

## 目录说明（核心）

- 训练脚本：`train_*.py`、`run_train_suite.py`、`run_kaggle_all.py`
- 模型/组件：`model_utils.py`、`*_classifier.py`、`*_densenet*.py`
- 数据处理：`split_data.py`、`prepare_kaggle_imagefolder.py`、`data_statistics.py`
- 配置：`config.py`
- 环境：`densenet_env_current.yml`

## 环境配置

优先使用 Conda 环境文件：

```bash
conda env create -f densenet_env_current.yml
conda activate <env_name>
```

## 数据集准备
数据来源：

```
IXI Dataset 
    https://brain-development.org/ixi-dataset/
        
Brain Tumor Dataset
    https://figshare.com/articles/dataset/brain_tumor_dataset/1512427     

```

需要将数据整理为 ImageFolder 结构（示例）：

```
data/
	train/
		class_a/
		class_b/
	val/
		class_a/
		class_b/
```

Kaggle 数据集可用 `prepare_kaggle_imagefolder.py` 预处理后放入 `data_kaggle_prepared/`（该目录默认忽略，不上传）。

## 运行示例

```bash
python train_resnet50.py
python train_vit_tiny.py
python run_train_suite.py
```

如果使用 VMamba/VSSD 等第三方实现：相关源码默认不入库（`external/` 已忽略），请自行克隆到 `external/` 目录（部分脚本运行时会给出缺失提示）。

