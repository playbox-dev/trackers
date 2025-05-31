import cv2
import torch
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from inference import get_model
from collections import deque
from trackers import DeepSORTTracker, ReIDModel
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import os



class GlobalIDAssigner:
    def __init__(self, feature_dim, sim_thresh=0.5, pool_size=20):
        self.next_id = 0
        self.feature_pools = {}
        self.sim_thresh = sim_thresh
        self.pool_size = pool_size

    def assign(self, detection_feats):
        assigned_ids = []
        for feat in detection_feats:
            # 1) 既存IDプールの「平均特徴」と比較
            best_id, best_sim = None, -1.0
            for gid, dq in self.feature_pools.items():
                # deque を numpy array にして axis=0で平均
                avg_feat = np.mean(np.stack(dq, axis=0), axis=0)
                # コサイン類似度
                sim = np.dot(avg_feat, feat) / (np.linalg.norm(avg_feat)*np.linalg.norm(feat)+1e-8)
                if sim > best_sim:
                    best_sim, best_id = sim, gid

            # 2) 類似度が閾値以上なら既存IDを更新、プールに append
            if best_sim >= self.sim_thresh:
                assigned_ids.append(best_id)
                self.feature_pools[best_id].append(feat)

            # 3) そうでなければ新IDを発行、そのプールを作成
            else:
                new_id = self.next_id
                self.next_id += 1
                dq = deque(maxlen=self.pool_size)
                dq.append(feat)
                self.feature_pools[new_id] = dq
                assigned_ids.append(new_id)

        return assigned_ids


class TorchReIDWrapper:
    def __init__(self, extractor):
        """
        extractor: torchreid.utils.FeatureExtractor
        """
        self.extractor = extractor

    def extract_features(self, detections, frame):
        """
        detections: sv.Detections (with .xyxy numpy array of shape [N,4])
        frame: np.ndarray image
        returns: np.ndarray of shape (N, feature_dim)
        """
        # 1) bbox を int に
        bboxes = detections.xyxy.astype(int)

        # 2) 各バウンディングボックスから画像を切り出し
        crops = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)

        # 3) torchreid extractor を呼び出す
        if len(crops) > 0:
            # 画像のリストを渡す
            feats = self.extractor(crops)
            # GPUテンソルをCPUに転送してからNumPy配列に変換
            return feats.cpu().numpy()
        else:
            # 検出がない場合は空の配列を返す
            return np.array([])


def compute_spatial_cost_matrix(dets1, dets2, H1, H2):
    """
    dets1, dets2: sv.Detections 各カメラの Detections オブジェクト
    H1, H2: np.ndarray (3×3)  各カメラ→ワールド平面へのホモグラフィ
    """
    if len(dets1) == 0 or len(dets2) == 0:
        return np.array([])

    # ワールド座標に射影した後の底辺中央 (x, y)
    def to_world(dets, H):
        pts = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            # bbox の下辺中央を homogeneous で
            p = np.array([(x1+x2)/2, y2, 1.0])
            pw = H @ p
            pw = pw[:2] / pw[2]
            pts.append(pw)
        return np.stack(pts, axis=0)

    world1 = to_world(dets1, H1)
    world2 = to_world(dets2, H2)

    OFFICE_HEIGHT, OFFICE_WIDTH = 943, 806
    OFFICE_DIAGONAL = np.hypot(OFFICE_HEIGHT, OFFICE_WIDTH)

    cost_matrix = np.zeros((len(dets1), len(dets2)), dtype=float)
    for i, p1 in enumerate(world1):
        for j, p2 in enumerate(world2):
            cost_matrix[i, j] = np.linalg.norm(p1 - p2) / OFFICE_DIAGONAL

    return cost_matrix


# --- 特徴量距離行列計算 ---
def compute_feature_cost_matrix(dets1, dets2, wrapped_reid, frame1, frame2):
    """
    特徴量の距離行列を計算
    """
    if len(dets1) == 0 or len(dets2) == 0:
        return np.array([])

    # 特徴抽出
    # feats1 = reid_model.extract_features(dets1, frame1)
    # feats2 = reid_model.extract_features(dets2, frame2)

    feats1 = wrapped_reid.extract_features(dets1, frame1)
    feats2 = wrapped_reid.extract_features(dets2, frame2)

    # コサイン距離行列を計算
    cost = np.zeros((len(dets1), len(dets2)), dtype=float)
    for i, feat1 in enumerate(feats1):
        for j, feat2 in enumerate(feats2):
            # コサイン類似度を距離に変換 (1 - 類似度)
            sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
            cost[i, j] = 1.0 - sim

    return cost


# --- 再IDマッチング ---
def associate_detections(dets1, dets2, H1, H2, wrapped_reid, frame1, frame2,
                         emb_thresh=0.5, spatial_thresh=0.2, alpha=0.7):
    """
    dets1/dets2: sv.Detections
    H1/H2: ホモグラフィ行列
    reid_model: 特徴抽出モデル
    alpha: 特徴量コストの重み (1-alpha が空間コストの重み)
    """
    if len(dets1) == 0 or len(dets2) == 0:
        return []

    # 特徴量コスト
    Cf = compute_feature_cost_matrix(dets1, dets2, wrapped_reid, frame1, frame2)
    if Cf.size == 0:
        return []

    # 空間コスト
    Cs = compute_spatial_cost_matrix(dets1, dets2, H1, H2)
    if Cs.size == 0:
        return []

    # 重み付き結合コスト
    C = alpha * Cf + (1 - alpha) * Cs

    # 閾値を超えるコストは大きな値に設定（np.infではなく）
    valid = (Cf < emb_thresh) & (Cs < spatial_thresh)
    C[~valid] = 1000.0  # np.infの代わりに大きな値を使用

    # 有効なマッチングが1つもない場合は早期リターン
    if not np.any(valid):
        return []

    # ハンガリアンアルゴリズムでマッチング
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(C)

        # 有効なマッチングのみ返す
        matches = []
        for i, j in zip(row_ind, col_ind):
            if valid[i, j]:  # 閾値を満たすマッチングのみ
                matches.append((i, j, C[i, j]))

        return matches
    except ValueError as e:
        print(f"Warning: {e}. Skipping matching for this frame.")
        return []


# --- オフィスマップへの描画関数 ---
def draw_on_office_map(office_map, detections_list, H_list, colors=None):
    """
    office_map: オフィスの平面図画像
    detections_list: 各カメラのDetectionsのリスト
    H_list: 各カメラのホモグラフィ行列のリスト
    colors: IDごとの色のマッピング辞書
    """
    result = office_map.copy()

    if colors is None:
        # ランダムな色を生成
        colors = {}

    # 各カメラの検出を描画
    for cam_idx, (dets, H) in enumerate(zip(detections_list, H_list)):
        if len(dets) == 0:
            continue

        for i in range(len(dets)):
            if not hasattr(dets, 'tracker_id'):
                continue

            track_id = dets.tracker_id[i]

            # IDごとに色を割り当て
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            color = colors[track_id]

            # 足元の位置をワールド座標に変換
            x1, y1, x2, y2 = dets.xyxy[i]
            foot_point = np.array([(x1+x2)/2, y2, 1.0])
            world_point = H @ foot_point
            world_point = (world_point[:2] / world_point[2]).astype(int)

            # 円と ID を描画
            cv2.circle(result, tuple(world_point), 10, color, -1)
            cv2.putText(result, f"ID:{track_id}",
                       (world_point[0] + 10, world_point[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result, colors


# --- 鳥瞰図軌跡描画関数 ---
def draw_bird_eye_view(office_map, detections_list, H_list, track_history, max_history=30, colors=None):
    """
    office_map: オフィスの平面図画像
    detections_list: 各カメラのDetectionsのリスト
    H_list: 各カメラのホモグラフィ行列のリスト
    track_history: IDごとの軌跡履歴 {id: deque([(x, y), ...]), ...}
    max_history: 軌跡の最大履歴数
    colors: IDごとの色のマッピング辞書
    """
    result = office_map.copy()

    if colors is None:
        # ランダムな色を生成
        colors = {}

    # 各カメラの検出を処理
    for cam_idx, (dets, H) in enumerate(zip(detections_list, H_list)):
        if len(dets) == 0 or not hasattr(dets, 'tracker_id'):
            continue

        for i in range(len(dets)):
            track_id = dets.tracker_id[i]

            # IDごとに色を割り当て
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            color = colors[track_id]

            # 足元の位置をワールド座標に変換
            x1, y1, x2, y2 = dets.xyxy[i]
            foot_point = np.array([(x1+x2)/2, y2, 1.0])
            world_point = H @ foot_point
            world_point = (world_point[:2] / world_point[2]).astype(int)

            # 軌跡履歴を更新
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=max_history)
            track_history[track_id].append(tuple(world_point))

            # 円と ID を描画
            cv2.circle(result, tuple(world_point), 10, color, -1)
            cv2.putText(result, f"ID:{track_id}",
                       (world_point[0] + 10, world_point[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 軌跡を描画
            points = list(track_history[track_id])
            if len(points) > 1:
                for j in range(1, len(points)):
                    # 線の太さを徐々に細くする（新しい点ほど太く）
                    thickness = max(1, int(3 * (j / len(points))))
                    cv2.line(result, points[j-1], points[j], color, thickness)

    return result, colors, track_history


def create_horizontal_layout(frames, bird_eye_view):
    """
    frames: カメラからの映像のリスト
    bird_eye_view: 鳥瞰図視点の軌跡画像

    3つの映像を横に並べるレイアウトを作成:
    | カメラ1 | カメラ2 | 鳥瞰図 |
    """
    # カメラ映像を横に連結
    camera_combined = cv2.hconcat(frames)

    # 鳥瞰図のサイズをカメラ映像の高さに合わせる
    h_cam = frames[0].shape[0]
    w_cam = frames[0].shape[1]

    # 鳥瞰図をリサイズ（高さをカメラ映像に合わせる）
    h_bird, w_bird = bird_eye_view.shape[:2]
    scale = h_cam / h_bird
    new_width = int(w_bird * scale)
    resized_bird_eye = cv2.resize(bird_eye_view, (new_width, h_cam))

    # 3つの映像を横に連結
    final_layout = cv2.hconcat([camera_combined, resized_bird_eye])

    return final_layout


def is_streaming_source(source_path):
    """
    入力ソースがストリーミング映像かどうかを判定する

    Args:
        source_path: 入力ソースのパスまたはURL

    Returns:
        bool: ストリーミング映像の場合True、それ以外はFalse
    """
    # RTSPストリーム
    if source_path.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        return True

    # IPカメラ（数字のみの場合はカメラインデックスと判断）
    if source_path.isdigit():
        return True

    # その他のストリーミングソース（必要に応じて追加）
    # ...

    return False


# --- マルチカメラ処理 ---
def multi_camera_tracking(source_videos, output_path, office_map_path, homography_matrices):
    # 出力ディレクトリを自動作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    # 1) 共通コンポーネント準備
    # model = YOLO("yolo11s.pt")
    # model = RFDETRBase()
    model = get_model("rfdetr-base")
    extractor = FeatureExtractor(
        model_name='osnet_x0_25',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        image_size=(256, 128),
    )
    wrapped_reid = TorchReIDWrapper(extractor)
    tracker_list = [DeepSORTTracker(reid_model=wrapped_reid) for _ in source_videos]

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT,
        text_scale=0.7,
        text_padding=3
    )
    global_id_assigner = GlobalIDAssigner(
        feature_dim=512,
        sim_thresh=0.5,
        pool_size=10,
    )

    # オフィスマップの読み込み
    office_map = cv2.imread(office_map_path)
    if office_map is None:
        raise RuntimeError(f"Failed to read office map from {office_map_path}")

    # ホモグラフィ行列の読み込み
    H_list = homography_matrices

    # 2) VideoCapture 準備
    caps = [cv2.VideoCapture(p) for p in source_videos]
    ret0, frame0 = caps[0].read()
    if not ret0:
        raise RuntimeError(f"Failed to read from {source_videos[0]}")
    h, w = frame0.shape[:2]
    # fps = caps[0].get(cv2.CAP_PROP_FPS)

    # 入力ソースの種類に応じてFPSを設定
    is_streaming = any(is_streaming_source(p) for p in source_videos)
    if is_streaming:
        # ストリーミング映像の場合は固定FPS
        fps = 10.0
        print(f"ストリーミング映像を検出: FPSを{fps}に設定")
    else:
        # 通常の動画ファイルの場合は元のFPSを使用
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        print(f"通常の動画ファイルを検出: FPS={fps}")
    n_cams = len(caps)

    # 3) 出力用 VideoWriter を作成
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # オフィスマップのサイズを取得
    map_h, map_w = office_map.shape[:2]

    # 2x2グリッドレイアウトのサイズを計算
    grid_width = max(w * n_cams, map_w * 2)
    grid_height = h + map_h

    writer = None

    # ID ごとの色マッピングと軌跡履歴
    id_colors = {}
    track_history = {}

    frame_count = 0
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frames = None
                break
            frames.append(frame)
        if frames is None:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # 各カメラで検出→トラッカー更新
        all_feats, all_dets, all_local_ids = [], [], []
        for idx, frame in enumerate(frames):
            # 1) モデル推論（1フレームあたり1つの ObjectDetectionInferenceResponse が返る）
            response_list = model.infer(frame, confidence=0.5)

            # 2) 万一、推論結果リストが空なら Detections.empty() を使う
            if not response_list:
                dets = sv.Detections.empty()
                all_dets.append(dets)
                continue

            # 3) 推論結果オブジェクトを取り出す
            response = response_list[0]  # ObjectDetectionInferenceResponse 型

            # 4) 内部にある 'predictions' リストを取り出す
            #    これが ObjectDetectionPrediction のリストになる
            all_preds = response.predictions  # list[ObjectDetectionPrediction, ...]

            # 5) その中から「class_id が 1 ＝ person（人）」のものだけを抽出
            #    ※COCOでは person の class_id が 1（あなたの出力を見た限り）
            person_preds = [
                p for p in all_preds
                if getattr(p, "class_id", None) == 1
            ]

            # 6) 抽出した人クラスのリスト（person_preds）を
            #     BoxAnnotator やトラッカーが使える形式に直す必要がある
            #
            #    Supervision の sv.Detections の場合、以下のように
            #    ・xyxy（左上座標・右下座標をまとめた NumPy 配列）
            #    ・confidence（NumPy 配列）
            #    ・class_id （NumPy 配列）
            #    を手動で作成してやる必要がある。
            import numpy as np

            # 6-1) xyxy（バウンディングボックスを [x1, y1, x2, y2] に変換）
            #       p.x, p.y は中心座標、p.width, p.height は幅・高さなので、
            #       左上= (x - w/2, y - h/2)、右下= (x + w/2, y + h/2)
            xyxy = np.array([
                [
                    p.x - p.width / 2,
                    p.y - p.height / 2,
                    p.x + p.width / 2,
                    p.y + p.height / 2,
                ]
                for p in person_preds
            ])

            # 6-2) confidence と class_id も NumPy 配列にする
            confidence = np.array([p.confidence for p in person_preds])
            class_id = np.array([p.class_id for p in person_preds])

            # 6-3) sv.Detections インスタンスを生成
            dets = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

            # 7) トラッカー（DeepSORT など）に渡して更新を行う
            dets = tracker_list[idx].update(dets, frame)

            # 8) 最終的な検出結果（人のみトラッキング済み）をリストに追加
            all_dets.append(dets)


            # 特徴ベクトルを抽出
            if len(dets) > 0:
                feats = wrapped_reid.extract_features(dets, frame)
                all_feats.append(feats)

            # ローカルIDを保存
            all_local_ids.append(dets.tracker_id.copy() if hasattr(dets, 'tracker_id') else np.array([]))

        # グローバルID 割り当て（特徴ベクトルがある場合のみ）
        if all_feats and any(f.size > 0 for f in all_feats):
            # 空でない特徴ベクトルを連結
            non_empty_feats = []
            feat_indices = []
            start_idx = 0

            for i, feats in enumerate(all_feats):
                if feats.size > 0:
                    non_empty_feats.append(feats)
                    n = len(feats)
                    feat_indices.append((i, start_idx, start_idx + n))
                    start_idx += n

            if non_empty_feats:
                stacked_feats = np.vstack(non_empty_feats)
                global_ids = global_id_assigner.assign(stacked_feats)

                # グローバルIDを各検出に割り当て
                for cam_idx, start, end in feat_indices:
                    if len(all_dets[cam_idx]) > 0:
                        all_dets[cam_idx].tracker_id = np.array(global_ids[start:end], dtype=int)

        # 空間的制約を利用したID同期（2台のカメラの場合）
        if len(all_dets) >= 2 and len(H_list) >= 2:
            for i in range(len(all_dets) - 1):
                for j in range(i + 1, len(all_dets)):
                    if len(all_dets[i]) > 0 and len(all_dets[j]) > 0:
                        try:
                            matches = associate_detections(
                                all_dets[i], all_dets[j],
                                H_list[i], H_list[j],
                                wrapped_reid, frames[i], frames[j],
                                emb_thresh=0.4,
                                spatial_thresh=0.15,
                                alpha=0.7
                            )

                            # マッチングに基づいてIDを同期
                            for idx_i, idx_j, cost in matches:
                                # カメラjのIDをカメラiのIDに合わせる
                                id_i = all_dets[i].tracker_id[idx_i]
                                all_dets[j].tracker_id[idx_j] = id_i

                                # 特徴プールも更新
                                if id_i in global_id_assigner.feature_pools:
                                    feat_j = wrapped_reid.extract_features(
                                        all_dets[j][idx_j:idx_j+1], frames[j]
                                    )
                                    if feat_j.size > 0:
                                        global_id_assigner.feature_pools[id_i].append(feat_j[0])
                        except Exception as e:
                            print(f"Error during ID synchronization: {e}")
                            continue

        # 重複IDの解消
        for cam_idx, dets in enumerate(all_dets):
            if len(dets) == 0 or not hasattr(dets, 'tracker_id'):
                continue

            ids, counts = np.unique(dets.tracker_id, return_counts=True)
            for gid, cnt in zip(ids, counts):
                if cnt > 1:
                    # このフレーム内で gid が重複
                    idxs = np.where(dets.tracker_id == gid)[0]
                    confs = getattr(dets, "confidence", np.zeros(len(dets)))
                    sorted_idxs = idxs[np.argsort(-confs[idxs])]

                    # 最もスコアの高い one を除き
                    for dup_idx in sorted_idxs[1:]:
                        new_id = global_id_assigner.next_id
                        global_id_assigner.next_id += 1
                        dets.tracker_id[dup_idx] = new_id

                        # 新ID 用の特徴プールも作っておく
                        feat = wrapped_reid.extract_features(dets[dup_idx:dup_idx+1], frames[cam_idx])
                        if feat.size > 0:
                            dq = deque(maxlen=global_id_assigner.pool_size)
                            dq.append(feat[0])
                            global_id_assigner.feature_pools[new_id] = dq

        # オフィスマップに検出を描画
        map_with_tracks, id_colors = draw_on_office_map(office_map, all_dets, H_list, id_colors)

        # 鳥瞰図視点の軌跡を描画
        bird_eye_view, id_colors, track_history = draw_bird_eye_view(
            office_map, all_dets, H_list, track_history, max_history=30, colors=id_colors
        )

        annotated = []
        for frame, dets, local_ids in zip(frames, all_dets, all_local_ids):
            confidences = getattr(dets, "confidence", None)
            if confidences is None:
                confidences = np.zeros(len(dets))

            # 各検出ごとにラベルを作成
            labels = []
            for i in range(len(dets)):
                if hasattr(dets, 'tracker_id'):
                    g_id = dets.tracker_id[i]
                    l_id = local_ids[i] if i < len(local_ids) else -1
                    conf = confidences[i]
                    labels.append(f"G{g_id} L{l_id} {conf:.2f}")

            # 矩形とラベルを描画
            f = box_annotator.annotate(frame, dets)
            f = label_annotator.annotate(f, dets, labels=labels)
            annotated.append(f)

        # 3つの映像を横に並べるレイアウトを作成
        horizontal_layout = create_horizontal_layout(annotated, bird_eye_view)

        # writer は最初の１回だけ初期化
        if writer is None:
            h_layout, w_layout = horizontal_layout.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (w_layout, h_layout)   # ← (幅, 高さ) の順番に注意
            )
            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter の初期化に失敗: size={(w_layout, h_layout)}")

        # フレームを書き出し
        writer.write(horizontal_layout)

    # 後片付け
    for cap in caps:
        cap.release()
    writer.release()
    print(f"Tracking completed. Output saved to {output_path}")


if __name__ == "__main__":
    # 入力ビデオパス
    source_videos = [
        "../data/output_bookshelf_20241025_110713_trimmed.mp4",
        "../data/output_trashcan_20241025_110713_trimmed.mp4",
        # "rtmp://192.168.0.106:1935/fairy_1",
        # "rtmp://192.168.0.106:1935/fairy_2",
    ]

    # 出力ビデオパス
    output_path = "../results/v2_stream_sample_person.mp4"

    # オフィスマップのパス
    office_map_path = "../data/office.png"

    # ホモグラフィ行列（カメラからオフィスマップへの変換）
    # 注: 実際のホモグラフィ行列に置き換える必要があります
    H1 = np.array([
        [0.1, 0.01, -50],
        [0.01, 0.1, -30],
        [0.0001, 0.0001, 1]
    ])

    H2 = np.array([
        [0.1, -0.01, 100],
        [0.01, 0.1, -30],
        [0.0001, -0.0001, 1]
    ])
    # ホモグラフィ行列をnpyファイルから読み込む
    H1_path = "../data/homography/homography_bookshelf_to_blueprint.npy"
    H2_path = "../data/homography/homography_trashcan_to_blueprint.npy"
    H1 = np.load(H1_path, allow_pickle=True)
    H2 = np.load(H2_path, allow_pickle=True)
    homography_matrices = [H1, H2]

    # 出力ディレクトリを自動作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    # トラッキング実行
    multi_camera_tracking(source_videos, output_path, office_map_path, homography_matrices)
