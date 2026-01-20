import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.missed = 0

    def update(self, bbox):
        self.bbox = bbox
        self.missed = 0


class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2]
        """
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return self.tracks

        cost = np.zeros((len(self.tracks), len(detections)))
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i, j] = 1 - iou(trk.bbox, det)

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if 1 - cost[r, c] >= self.iou_threshold:
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1

        return self.tracks
