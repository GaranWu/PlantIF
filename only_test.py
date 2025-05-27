import os
import numpy as np
import torch
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader

from config import get_config
from solver_test import Solver
from utils.custom_dataset import CustomDataset


def set_random_seed(seed=336):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == '__main__':
    set_random_seed()

    # 加载测试配置
    test_config = get_config(mode='test')
    print("[INFO] Test config loaded.")

    # 图像和文本处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        local_files_only=True
    )

    # 构建测试数据集
    test_dataset = CustomDataset(
        "/home/wuxingcai/wxc/PlantMIT/datasets/522/all_data_filtered.csv",
        "/home/wuxingcai/wxc/PlantMIT/datasets/522/",
        transform,
        tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 初始化 Solver
    solver = Solver(
        train_config=test_config,  # 用 test_config 填 train_config，因构造函数需要它
        dev_config=None,
        test_config=test_config,
        train_data_loader=None,
        dev_data_loader=None,
        test_data_loader=test_loader,
        is_train=False
    )

    # 构建模型并加载训练好的权重
    solver.build()

    # 手动加载 checkpoint 权重
    model_path = f'checkpoints/model_2025-05-06_16:30:01.std'
    if os.path.exists(model_path):
        print(f"[INFO] Loading model from {model_path}")
        solver.model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # 执行测试
    print("[INFO] Running evaluation on test set...")
    test_loss, test_acc = solver.eval(mode="test", to_print=True)
    print(f"[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
