'''
iou追踪示例
'''
import csv

import pandas as pd
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO
import cv2
import numpy as np



class IouTracker:
    def __init__(self):
        self.max_missed_frames = 0
        self.sigma_mahalanobis = 10
        # "F:\yolo\fish\1\运动（速度慢-快）\30s 慢-快\CYC\CK1 30s.mp4"
        self.data=r"F:\鱼的数据\剪辑版\P-10A2-MAH00034.mp4"
        self.detection_model = YOLO(r"D:\yolov10\runs\wang\weights\best.pt")
        self.objs_labels = self.detection_model.names
        self.track_classes = {0: 'fish'}
        self.iou_threshold=0.03
        self.conf_thresh = 0.1
        self.frame=None
        self.pre_frame=None
        self.cost_threshold=500
        self.iou_values = []
    def analyze_iou_values(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from scipy.ndimage import gaussian_filter1d

        # 先将每一帧的所有 IOU 值收集起来
        iou_values = [data[0] for data in self.iou_values]
        frame_ids = [data[1] for data in self.iou_values]

        # 使用 defaultdict 来按帧分组 IOU 值
        frame_iou_map = defaultdict(list)

        # 将 IOU 值按 frame_id 分组
        for iou, frame_id in zip(iou_values, frame_ids):
            frame_iou_map[frame_id].append(iou)

        # 计算每一帧的平均 IOU
        avg_iou_per_frame = {frame_id: np.mean(iou_list) for frame_id, iou_list in frame_iou_map.items()}

        # 提取平均 IOU 和帧 ID
        avg_iou_values = list(avg_iou_per_frame.values())
        avg_frame_ids = list(avg_iou_per_frame.keys())

        # 对平均 IOU 值进行平滑处理（使用移动平均或高斯平滑）

        # 移动平均
        window_size = 5  # 可以调整窗口大小来控制平滑程度
        smoothed_iou_values_ma = np.convolve(avg_iou_values, np.ones(window_size) / window_size, mode='valid')

        # 高斯平滑（可以根据需要调整标准差）
        smoothed_iou_values_gaussian = gaussian_filter1d(avg_iou_values, sigma=2)

        # 计算整体统计信息
        mean_iou = np.mean(smoothed_iou_values_gaussian)
        median_iou = np.median(smoothed_iou_values_gaussian)
        std_iou = np.std(smoothed_iou_values_gaussian)

        print(f"Total IOU values collected: {len(iou_values)}")
        print(f"Mean IOU (smoothed over frames): {mean_iou:.4f}")
        print(f"Median IOU (smoothed over frames): {median_iou:.4f}")
        print(f"Standard Deviation (smoothed over frames): {std_iou:.4f}")

        # 画出每一帧的平均 IOU 随时间变化的图，使用平滑后的数据
        plt.figure(figsize=(10, 6))
        plt.plot(avg_frame_ids[:len(smoothed_iou_values_ma)], smoothed_iou_values_ma, color='blue', marker='o',
                 markersize=4, linestyle='-', linewidth=2, label='Moving Average')
        plt.plot(avg_frame_ids, smoothed_iou_values_gaussian, color='red', linestyle='-', linewidth=2,
                 label='Gaussian Smoothing')
        plt.title('Smoothed Average IOU Value Over Time')
        plt.xlabel('Frame ID')
        plt.ylabel('Smoothed Average IOU Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 绘制 IOU 分布的直方图
        plt.figure(figsize=(10, 6))
        plt.hist(iou_values, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of IOU Values')
        plt.xlabel('IOU Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # 提供一个 IOU 阈值建议
        percentile = 75
        suggested_threshold = np.percentile(iou_values, percentile)
        print(f"Suggested IOU threshold at {percentile}th percentile: {suggested_threshold:.4f}")
    def calculate_iou(self, bbox1, bbox2):
        """

        计算两个bounding box的IOU

        """
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0.0

        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union
    def black_calculate_iou(self, bboxt, bboxd):
        """
        计算两个bounding box的black_IOU
        """
        # 追踪框
        x1_t, y1_t, x2_t, y2_t = map(int, bboxt)
        roi_track = self.pre_frame[y1_t:y2_t, x1_t:x2_t]
        gray_track = cv2.cvtColor(roi_track, cv2.COLOR_BGR2GRAY)
        _, binary_track = cv2.threshold(gray_track, 112, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        t_eroded_roi = cv2.erode(binary_track, kernel, iterations=1)
        # 膨胀操作
        t_dilated_roi = cv2.dilate(t_eroded_roi, kernel, iterations=1)
        t_black_pixels = np.column_stack(np.where(t_dilated_roi == 0)) + np.array([y1_t, x1_t])
        # 检测框
        x1_t, y1_t, x2_t, y2_t = map(int, bboxd)  # 追踪框
        roi_track = self.frame[y1_t:y2_t, x1_t:x2_t]
        gray_track = cv2.cvtColor(roi_track, cv2.COLOR_BGR2GRAY)
        _, binary_track = cv2.threshold(gray_track, 112, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        d_eroded_roi = cv2.erode(binary_track, kernel, iterations=1)
        d_dilated_roi = cv2.dilate(d_eroded_roi, kernel, iterations=1)
        d_black_pixels = np.column_stack(np.where(d_dilated_roi == 0)) + np.array([y1_t, x1_t])

        (x0_1, y0_1, x1_1, y1_1) = bboxt
        (x0_2, y0_2, x1_2, y1_2) = bboxd
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0.0

        dt = np.dtype([('x', t_black_pixels.dtype), ('y', t_black_pixels.dtype)])
        t_struct = t_black_pixels.view(dt).reshape(-1)
        d_struct = d_black_pixels.view(dt).reshape(-1)
        common_pixels_struct = np.intersect1d(t_struct, d_struct)
        # 转换回二维坐标数组
        common_pixels = common_pixels_struct.view(t_black_pixels.dtype).reshape(-1, 2)
        # intersection_img = np.ones_like(self.frame, dtype=np.uint8) * 255  # 创建一个全白的图像
        # intersection_img2 = np.ones_like(self.frame, dtype=np.uint8) * 255  # 创建一个全白的图像
        # intersection_img3 = np.ones_like(self.frame, dtype=np.uint8) * 255  # 创建一个全白的图像
        # save_dir = r"D:\yolov10\ultralytics-main\runs"
        # for pixel in t_black_pixels:
        #     y, x = pixel
        #     intersection_img2[y, x] = 0  # 设置为黑色
        # for pixel in d_black_pixels:
        #     y, x = pixel
        #     intersection_img3[y, x] = 0  # 设置为黑色
        # # 将交集的像素设置为黑色
        # for pixel in common_pixels:
        #     y, x = pixel
        #     intersection_img[y, x] = 0  # 设置为黑色
        # cv2.imwrite(os.path.join(save_dir, "intersection——max.png"),  intersection_img2)
        # cv2.imwrite(os.path.join(save_dir, "intersection——dil.png"), intersection_img3)
        # cv2.imwrite(os.path.join(save_dir, "intersection3.png"), intersection_img)

        intersection = len(common_pixels)
        union = len(t_black_pixels) + len(d_black_pixels) - intersection
        iou = intersection / union if union > 0 else 0
        # cv2.imshow("intersection_img",intersection_img)
        # cv2.waitKey(0)
        return iou
    def predict(self, frame):
        '''
        检测
        '''
        # model = YOLO(frame)
        # results = model.track(frame, save=True, conf=0.1)
        result = list(self.detection_model(frame, stream=True, conf=self.conf_thresh))[
            0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array

        dets = []

        for box in boxes.data:
            l, t, r, b = box[:4]  # left, top, right, bottom
            conf, class_id = box[4:]  # confidence, class

            if class_id not in self.track_classes:
                continue
            dets.append({'bbox': [l, t, r, b], 'score': conf, 'class_id': class_id})
        return dets

    def mahalanobis_distance(self, track_center, det_center, cov_matrix):
        diff = np.array(track_center) - np.array(det_center)
        inv_cov_matrix = np.linalg.inv(cov_matrix)  # 协方差矩阵的逆
        return np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))

    def main(self):
        # 读取视频
        cap = cv2.VideoCapture(self.data)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        some_weight=1
        center_points = {}
        tracks_active = []
        frame_id = 1
        min_track_id = 1
        self.pre_frame=None
        out = cv2.VideoWriter("test_out-p2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
        max_track_id = 10
        frame_data = []
        while True:
            ret, raw_frame = cap.read()
            if ret:
                self.frame = cv2.resize(raw_frame, (1280, 720))
                raw_frame = self.frame
                dets = self.predict(raw_frame)
                for track in tracks_active:
                    if len(dets) > 0:
                        track_bbox = track['bboxes'][-1]
                        track_center = [(track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2]
                        cov_matrix = np.array([[1, 0], [0, 1]])
                        candidate_dets = []
                        for det in dets:
                            det_center = [
                                (det['bbox'][0] + det['bbox'][2]) / 2,
                                (det['bbox'][1] + det['bbox'][3]) / 2
                            ]
                            mahalanobis_dist = self.mahalanobis_distance(
                                track_center, det_center, cov_matrix
                            )
                            iou = self.black_calculate_iou(track_bbox, det['bbox'])
                            if mahalanobis_dist < self.sigma_mahalanobis and iou > self.iou_threshold:
                                candidate_dets.append(det)
                                self.iou_values.append((iou,frame_id))  # Store IOU value
                        if candidate_dets:
                            best_match = max(
                                candidate_dets,
                                key=lambda det: self.black_calculate_iou(track_bbox, det['bbox'])
                            )
                            track['bboxes'].append(best_match['bbox'])
                            track['max_score'] = max(track['max_score'], best_match['score'])
                            track['last_seen_frame'] = frame_id
                            track['state'] = 'certain'
                            track['missed_frames'] = 0
                            dets.remove(best_match)
                        else:
                            track['missed_frames'] += 1
                            if track['missed_frames'] >= self.max_missed_frames:
                                track['state'] = 'uncertain'
                self.pre_frame = self.frame
                for det in dets:
                    if min_track_id > max_track_id:
                        break
                    new_track = {
                        'bboxes': [det['bbox']],
                        'max_score': det['score'],
                        'start_frame': frame_id,  # 初始检测到的帧数
                        'track_id': min_track_id,  # 使用编号为 track_idx 的检测器
                        'class_id': det['class_id'],
                        'last_seen_frame': frame_id,  # 新建时记录出现帧
                        'state': 'uncertain',  # 初始状态为不确定
                        'missed_frames': 0,
                    }
                    min_track_id += 1
                    tracks_active.append(new_track)
                # 漏检处理
                unconfirmed_tracks_last_frame = [track for track in tracks_active if frame_id - track['last_seen_frame'] > 0 ]
                confirmed_tracks_last_frame = [track for track in tracks_active if frame_id - track['last_seen_frame'] == 0 ]
                if len(unconfirmed_tracks_last_frame) > 0 and len(dets) > 0:
                    cost_matrix = np.zeros((len(unconfirmed_tracks_last_frame), len(dets)), dtype=np.float32)
                    # 构建成本矩阵
                    for i, track in enumerate(unconfirmed_tracks_last_frame):
                        # 获取轨迹的历史中心点
                        if len(track['bboxes']) >= 2:
                            # 使用最近两帧的中心点计算速度向量
                            prev_bbox = track['bboxes'][-2]
                            curr_bbox = track['bboxes'][-1]
                            prev_center = [
                                (prev_bbox[0] + prev_bbox[2]) / 2,
                                (prev_bbox[1] + prev_bbox[3]) / 2
                            ]
                            curr_center = [
                                (curr_bbox[0] + curr_bbox[2]) / 2,
                                (curr_bbox[1] + curr_bbox[3]) / 2
                            ]
                            # 计算速度向量
                            velocity = [
                                curr_center[0] - prev_center[0],
                                curr_center[1] - prev_center[1]
                            ]
                            # 预测下一个中心点位置
                            predicted_center = [
                                curr_center[0] + velocity[0],
                                curr_center[1] + velocity[1]
                            ]
                        else:
                            # 如果只有一个历史点，使用当前中心点作为预测
                            curr_bbox = track['bboxes'][-1]
                            predicted_center = [
                                (curr_bbox[0] + curr_bbox[2]) / 2,
                                (curr_bbox[1] + curr_bbox[3]) / 2
                            ]

                        for j, det in enumerate(dets):
                            det_bbox = det['bbox']
                            det_center = [
                                (det_bbox[0] + det_bbox[2]) / 2,
                                (det_bbox[1] + det_bbox[3]) / 2
                            ]
                            # 计算预测中心点与检测框中心点之间的欧氏距离
                            distance = np.linalg.norm(np.array(predicted_center) - np.array(det_center))
                            # 可选择结合IoU和距离计算成本
                            iou = self.calculate_iou(curr_bbox, det_bbox)
                            print("distance:",distance,"iou:",iou)
                            cost = distance - iou * 100000  # 定义一个合适的权重 some_weight
                            cost_matrix[i, j] = cost

                    # 匈牙利算法分配
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    for row, col in zip(row_indices, col_indices):
                        cost = cost_matrix[row, col]
                        if cost < self.cost_threshold:  # 定义一个合理的阈值
                            track = unconfirmed_tracks_last_frame[row]
                            det = dets[col]
                            # 更新轨迹
                            track['bboxes'].append(det['bbox'])
                            track['max_score'] = max(track['max_score'], det['score'])
                            track['last_seen_frame'] = frame_id
                            track['state'] = 'certain'
                            track['missed_frames'] = 0
                            print("匹配成功！！！",track["track_id"])
                            # dets.remove(det)
                        else:
                            # 如果成本过高，不进行分配
                            continue
                unconfirmed_tracks_last_frame = [track for track in tracks_active if frame_id - track['last_seen_frame'] > 0 ]
                confirmed_tracks_last_frame = [track for track in tracks_active if frame_id - track['last_seen_frame'] == 0 ]
                # 失配匹配
                for track in unconfirmed_tracks_last_frame:
                    # 获取当前丢失的检测器的上一帧
                    previous_frame_id = track['last_seen_frame']
                    # 获取上一帧的所有确定态检测器
                    last_tracks = [t for t in confirmed_tracks_last_frame]
                    max_iou = 0
                    best_det = None
                    for last_track in last_tracks:
                        # 遍历上一帧的所有检测框
                        # print(last_track)
                        iou = self.calculate_iou(last_track['bboxes'][-1],track['bboxes'][-1])
                        if iou > max_iou:
                            max_iou = iou
                            best_det = last_track
                    # 如果找到了匹配的检测框且 IOU 超过阈值
                    if best_det is not None and max_iou > 0:  # 设定一个适当的阈值
                        print(track["track_id"],"失配")
                        track['bboxes'].append(best_det['bboxes'][-1])
                        track['max_score'] = max(track['max_score'], best_det['max_score'])
                        track['state'] = 'uncertain'
                        track['missed_frames'] = 0
                        # 移除已匹配的检测框
                cross_line_color = (0, 255, 0)
                frame_rate = 30  # 根据实际情况设置
                max_points = frame_rate * 1  # 保留三秒的轨迹点
                # 画图
                for tracker in tracks_active:
                    l, t, r, b = tracker['bboxes'][-1]
                    l, t, r, b = int(l), int(t), int(r), int(b)
                    cx = int((l + r) / 2)
                    cy = int((t + b) / 2)
                    new_values = (cx, cy)
                    specific_key = tracker['track_id']
                    frame_data.append({
                        'frame_id': frame_id,
                        'track_id': specific_key,
                        'center_x': cx,
                        'center_y': cy
                    })
                    if tracker['state'] == 'certain':
                        if specific_key in center_points:
                            center_points[specific_key].append(new_values)
                            # 保留最近三秒的轨迹
                            if len(center_points[specific_key]) > max_points:
                                center_points[specific_key] = center_points[specific_key][-max_points:]
                        else:
                            center_points[specific_key] = [new_values]
                        # cv2.rectangle(raw_frame, (l, t), (r, b), cross_line_color, 2)
                        cv2.putText(raw_frame, f"{tracker['track_id']}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)
                        cv2.circle(raw_frame, (cx, cy), 3, (0, 0, 255), -1)

                for track_id, points_list in center_points.items():
                    for i in range(1, len(points_list)):
                        if points_list[i - 1] is None or points_list[i] is None:
                            continue
                        cv2.line(raw_frame, points_list[i - 1], points_list[i], (255, 0, 0), 2)
                print(frame_id)
                frame_id += 1  # 更新帧ID
                # 写入当前帧到输出视频
                # out.write(raw_frame)
                # cv2.imshow("Tracking", raw_frame)  # 添加实时显示当前帧
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
                    break
            else:
                break
        cap.release()
        out.release()
        self.analyze_iou_values()
        fieldnames = ['frame_id', 'track_id', 'center_x', 'center_y']
        with open(self.data+'tracking_data.csv', mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in frame_data:
                writer.writerow(data)
iou_tracker = IouTracker()
# 运行
iou_tracker.main()
