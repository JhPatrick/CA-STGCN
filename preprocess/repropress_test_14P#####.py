import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import argparse

# 配置参数
TEST_TXT_PATH = r"D:\python object\PoseToAction\dataset\test_14P.txt"
INPUT_VIDEO_DIR = r"D:\数据\数据集建立\SWAR"
OUTPUT_DIR = r"../dataset/sequences_14P_test_l"
MODEL_PATH = r"../yolo11l-pose.pt"
SELECTED_KPTS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 14个关键点

# 动作类别映射
CLASS_MAP = {
    '0': 'fall',
    '1': 'moving',
    '2': 'squatting',
    '3': 'squatting_operation',
    '4': 'standing',
    '5': 'standing_operation'
}


def process_video(video_path, model):
    """处理单个视频文件，提取关键点序列"""
    cap = cv2.VideoCapture(video_path)
    keypoint_sequences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLO进行关键点检测
        results = model(frame, verbose=False)[0]

        if len(results.boxes.conf) > 0:
            # 取置信度最高的人体检测结果
            max_conf_idx = torch.argmax(results.boxes.conf).item()
            kpts = results.keypoints[max_conf_idx]
            kpts_xy = kpts.xyn.cpu().numpy().squeeze()
            kpts_conf = kpts.conf.cpu().numpy().squeeze()

            # 构建完整关键点
            full_kpts = np.zeros((17, 3))
            full_kpts[:, :2] = kpts_xy
            full_kpts[:, 2] = kpts_conf
        else:
            # 当视频帧中未检测到任何人时，用全零填充
            full_kpts = np.zeros((17, 3))

        # 选择所需的14个关键点
        selected_kpts = full_kpts[SELECTED_KPTS]
        keypoint_sequences.append(selected_kpts.flatten())

    cap.release()
    return np.array(keypoint_sequences)


def process_test_set(test_txt_path, input_video_dir, output_dir, model):
    """处理测试集视频文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取测试集文件
    with open(test_txt_path, 'r') as f:
        test_samples = f.readlines()

    # 统计进度
    processed_count = 0
    errors = []

    # 进度条
    pbar = tqdm(test_samples, desc="处理测试集视频")

    for line in pbar:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        # 解析序列文件路径和类别
        seq_path = parts[0]
        class_id = parts[1]

        # 从路径中提取文件名
        filename = os.path.basename(seq_path)

        # 根据类别映射获取类别名称
        class_name = CLASS_MAP.get(class_id)
        if not class_name:
            continue

        # 构建输入视频路径 (假设视频文件名与序列文件名前缀相同)
        video_name = os.path.splitext(filename)[0]  # 去除.npy扩展名
        video_path = os.path.join(input_video_dir, class_name, f"{video_name}.mp4")

        # 确保视频文件存在
        if not os.path.isfile(video_path):
            errors.append(f"视频文件不存在: {video_path}")
            continue

        # 创建类别输出目录
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        try:
            # 处理视频并提取关键点序列
            sequence = process_video(video_path, model)

            # 保存序列数据
            output_path = os.path.join(class_output_dir, filename)
            np.save(output_path, sequence)

            processed_count += 1
            pbar.set_description(f"处理测试集视频 - 已处理 {processed_count}/{len(test_samples)}")
        except Exception as e:
            errors.append(f"处理视频 {video_path} 时出错: {str(e)}")

    # 输出处理结果
    print(f"\n处理完成! 成功处理 {processed_count}/{len(test_samples)} 个测试样本")

    if errors:
        print("\n错误信息:")
        for error in errors:
            print(f" - {error}")
    else:
        print("所有测试样本处理成功")


if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="重新处理测试集视频")
    parser.add_argument("--test_txt", default=TEST_TXT_PATH, help="测试集文本文件路径")
    parser.add_argument("--input_dir", default=INPUT_VIDEO_DIR, help="原始视频目录")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="输出序列目录")
    parser.add_argument("--model_path", default=MODEL_PATH, help="YOLO模型路径")
    args = parser.parse_args()

    # 加载YOLO模型
    print(f"加载YOLO模型: {args.model_path}")
    model = YOLO(args.model_path)

    # 处理测试集
    process_test_set(
        test_txt_path=args.test_txt,
        input_video_dir=args.input_dir,
        output_dir=args.output_dir,
        model=model
    )