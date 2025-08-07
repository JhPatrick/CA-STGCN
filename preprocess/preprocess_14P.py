import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# 配置参数
INPUT_DIR = r"D:\数据\数据集建立\SWAR"
OUTPUT_DIR = r"D:\python object\PoseToAction\dataset\sequences_14P"
SELECTED_KPTS = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]  # 14个关键点

# 选取YOLO-Pose的14个关键点
YOLO_KEYPOINTS = {
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

def process_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
    keypoint_sequences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # 处理检测结果
        if len(results.boxes.conf) > 0:
            max_conf_idx = torch.argmax(results.boxes.conf).item()
            kpts = results.keypoints[max_conf_idx]
            kpts_xy = kpts.xyn.cpu().numpy().squeeze()
            kpts_conf = kpts.conf.cpu().numpy().squeeze()

            # 构建完整关键点 (17个点)
            full_kpts = np.zeros((17, 3))
            full_kpts[:, :2] = kpts_xy
            full_kpts[:, 2] = kpts_conf
        else:
            # 无人时填充全零
            full_kpts = np.zeros((17, 3))

        selected_kpts = full_kpts[SELECTED_KPTS]
        keypoint_sequences.append(selected_kpts.flatten())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(keypoint_sequences))
    cap.release()

if __name__ == "__main__":
    # 全局加载模型
    model = YOLO('../yolo11x-pose.pt')  # 确认路径正确

    for class_name in os.listdir(INPUT_DIR):
        class_dir = os.path.join(INPUT_DIR, class_name)

        # 跳过非目录文件
        if not os.path.isdir(class_dir):
            continue

        video_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.mp4', '.avi'))]

        for video_file in tqdm(video_files, desc=f"处理 {class_name}"):
            video_path = os.path.join(class_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(OUTPUT_DIR, class_name, f"{video_name}.npy")

            process_video(video_path, output_path, model)