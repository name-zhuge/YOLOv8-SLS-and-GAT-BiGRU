
import numpy as np

def extract_samples_from_concentration(X, seconds=10, frames_per_second=2, step_seconds=3):
    """
    从每个浓度类别的视频数据中提取样本。
    每个样本包含 10 秒的数据，每秒提取 2 帧，总共提取 20 帧。
    对于每个浓度类别的数据，步进为 3 秒。

    参数：
    - X: 每个浓度类别的特征矩阵列表，格式为 [(5520, 160), (5728, 160), ...]
    - seconds: 提取样本的秒数，默认 10 秒
    - frames_per_second: 每秒提取的帧数，默认 2 帧
    - step_seconds: 每次提取开始的时间步长，默认每 3 秒提取一次

    返回：
    - all_samples: 提取的所有样本，形状为 (num_samples, 20, 160)
    - sample_counts: 每个视频提取的样本数量，格式为 [num_samples_video_1, num_samples_video_2, ...]
    """
    all_samples = []  # 用来存储所有浓度类别的样本
    sample_counts = []  # 用来存储每个视频的样本数量

    for concentration_data in X:
        num_frames = concentration_data.shape[0]  # 每个视频的帧数

        # 每次提取样本时的步长为step_seconds秒
        step_frames = step_seconds * frames_per_second  # 每次提取的帧数

        # 计算样本数量
        num_samples = 0
        ans = 0
        a = 0
        while ans * 90 + 284 <= num_frames:
            sample = []
            num = 0
            for second in range(seconds):
                frame_1 = step_seconds * 30 * ans + num * 30  # 第一帧位置
                frame_15 = frame_1 + 14  # 第15帧位置

                if frame_15 < num_frames:
                    sample.append(concentration_data[frame_1])  # 添加第一帧
                    sample.append(concentration_data[frame_15])  # 添加第15帧
                num += 1

            # 将提取的样本添加到结果中
            a += 1
            all_samples.append(np.array(sample))  # 转换为 NumPy 数组并添加到样本列表中
            ans += 1
            num_samples += 1  # 更新样本数量

        # 将当前视频的样本数量添加到样本数量列表中
        sample_counts.append(num_samples)

    # 转换 all_samples 为 NumPy 数组
    all_samples = np.array(all_samples)

    # 返回 NumPy 数组形式的 all_samples 和 sample_counts
    return all_samples, sample_counts
