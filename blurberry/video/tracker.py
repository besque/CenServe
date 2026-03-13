"""
Lightweight SORT tracker wrapper.
SORT uses Kalman filter + Hungarian algorithm to track bounding boxes across frames.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
from shared_types import DetectionEvent


def iou(bb_test, bb_gt):
    """Intersection over Union between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_gt = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / union if union > 0 else 0


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox, event_type, blur=True):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.1
        self.kf.Q[4:, 4:] *= 0.1
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(1, y2 - y1)
        self.kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.event_type = event_type
        self.blur = blur
        self.hits = 0
        self.age = 0

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(1, y2 - y1)
        self.kf.update(np.array([[cx], [cy], [s], [r]]))
        self.time_since_update = 0
        self.hits += 1

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        x = self.kf.x
        cx, cy, s, r = x[0,0], x[1,0], x[2,0], x[3,0]
        w = np.sqrt(abs(s * r))
        h = abs(s) / max(w, 1)
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    def get_bbox(self):
        return self.predict()


class MultiObjectTracker:
    def __init__(self, max_age=60, min_hits=1, iou_threshold=0.15):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[DetectionEvent]) -> List[DetectionEvent]:
        """
        Feed new detections. Returns active tracked boxes as DetectionEvents.
        Call this every frame — pass empty list on non-detection frames.
        """
        # Predict existing trackers
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())

        # Match detections to trackers via IoU
        matched_indices = set()
        if detections and predicted_boxes:
            iou_matrix = np.zeros((len(detections), len(predicted_boxes)))
            for d_idx, det in enumerate(detections):
                for t_idx, pred in enumerate(predicted_boxes):
                    iou_matrix[d_idx, t_idx] = iou(det.bbox, pred)
            
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self.trackers[c].update(detections[r].bbox)
                    self.trackers[c].blur = detections[r].blur
                    matched_indices.add(r)

        # Spawn new trackers for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_indices:
                self.trackers.append(
                    KalmanBoxTracker(det.bbox, det.type, det.blur)
                )

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Return active tracked boxes
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                results.append(DetectionEvent(
                    type=t.event_type,
                    bbox=t.get_bbox(),
                    confidence=0.9,
                    frame_id=-1,
                    blur=t.blur
                ))
        return results