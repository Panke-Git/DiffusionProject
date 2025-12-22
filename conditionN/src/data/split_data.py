"""
    @Project: UIR-PolyKernel
    @Author: Panke
    @FileName: split_data.py
    @Time: 2025/5/15 00:12
    @Email: None
"""
import os
import shutil
from sklearn.model_selection import train_test_split

def is_image_file(filename):
    """检查文件是否为图片格式"""
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def split_and_copy_dataset(src_dir, input_subdir, target_subdir, train_dir, val_dir, test_size=0.1, random_state=42):
    """
    划分数据集为训练集和验证集，并将其拷贝到指定目录
    :param src_dir: 数据集的根目录
    :param input_subdir: 输入图像子文件夹（如：'LSUI/input'）
    :param target_subdir: 目标图像子文件夹（如：'LSUI/GT'）
    :param train_dir: 训练集存放的目标路径
    :param val_dir: 验证集存放的目标路径
    :param test_size: 划分验证集的比例
    :param random_state: 随机种子，保证每次划分相同
    """

    # 获取所有图像文件路径
    input_files = sorted(os.listdir(os.path.join(src_dir, input_subdir)))
    target_files = sorted(os.listdir(os.path.join(src_dir, target_subdir)))

    input_paths = [os.path.join(src_dir, input_subdir, x) for x in input_files if is_image_file(x)]
    target_paths = [os.path.join(src_dir, target_subdir, x) for x in target_files if is_image_file(x)]

    # 确保输入和目标图像数目一致
    assert len(input_paths) == len(target_paths), "输入和目标图像数目不匹配！"

    # 使用 train_test_split 划分数据集
    train_inp, val_inp, train_tar, val_tar = train_test_split(
        input_paths, target_paths, test_size=test_size, random_state=random_state)

    # 拷贝文件到训练集和验证集的目录
    copy_files(train_inp, train_dir, 'input')
    copy_files(train_tar, train_dir, 'GT')
    copy_files(val_inp, val_dir, 'input')
    copy_files(val_tar, val_dir, 'GT')

    print(f"数据集划分并拷贝完成：\n训练集：{len(train_inp)} 文件\n验证集：{len(val_inp)} 文件")

def copy_files(file_list, dest_dir, subfolder):
    """将文件列表拷贝到目标文件夹，并按子文件夹划分"""
    if not os.path.exists(os.path.join(dest_dir, subfolder)):
        os.makedirs(os.path.join(dest_dir, subfolder))

    for file in file_list:
        shutil.copy(file, os.path.join(dest_dir, subfolder, os.path.basename(file)))


if __name__ == '__main__':
    # 示例使用
    src_dir = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\UIEB'  # 替换为你的数据集路径
    input_subdir = 'input'
    target_subdir = 'GT'

    train_dir = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\UIEB19\Train' # 替换为你的训练集保存路径
    val_dir = r'E:\PythonProject\01_Personal\UnderwaterImageEnhanced\dataset\UIEB19\Val'# 替换为你的验证集保存路径

    split_and_copy_dataset(src_dir, input_subdir, target_subdir, train_dir, val_dir, test_size=0.1)

