import cv2
import os
import math
import numpy as np
from ultralytics import YOLO
from movement_tracker import CrowdMovementClassifier


COLOR_WALKING  = (0, 220, 0)
COLOR_STANDING = (0, 60, 220)
COLOR_WARMUP   = (200, 200, 0)
COLOR_HUD_BG   = (20, 20, 20)
COLOR_HUD_TEXT = (240, 240, 240)


# -----------------------------------------------------------------------
# Camera motion estimator
#
# Uses estimateAffinePartial2D which handles ALL 2D camera motions:
#   pan left/right, tilt up/down, rotation, zoom in/out
#
# Returns:
#   M     — 2x3 affine matrix describing the full camera transform
#   dx,dy — translation part only (how many pixels camera slid sideways)
# -----------------------------------------------------------------------
def estimate_camera_transform(prev_gray, curr_gray):
    if prev_gray is None:
        return None, 0.0, 0.0

    # find strong corner points in the previous frame
    pts = cv2.goodFeaturesToTrack(prev_gray,
                                   maxCorners=300,
                                   qualityLevel=0.01,
                                   minDistance=8)
    if pts is None or len(pts) < 10:
        return None, 0.0, 0.0

    # track those points into the current frame
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)

    good_old = pts[status.flatten() == 1]
    good_new = new_pts[status.flatten() == 1]

    if len(good_old) < 6:
        return None, 0.0, 0.0

    # estimateAffinePartial2D with RANSAC:
    #   - handles pan, tilt, rotation, zoom all at once
    #   - RANSAC ignores outlier points (moving people) automatically
    M, _ = cv2.estimateAffinePartial2D(good_old, good_new,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=3)
    if M is None:
        return None, 0.0, 0.0

    # translation component is in the last column of M
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    return M, dx, dy


def warp_point(M, x, y):
    """
    Move a single point (x,y) using affine matrix M.
    Used to predict where an old track position moved to after camera moved.
    """
    if M is None:
        return x, y
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return int(nx), int(ny)


# -----------------------------------------------------------------------
# Simple ID tracker — with camera-aware matching
#
# Key fix: before matching detections to old tracks, we warp the old
# track positions using the camera transform M. This predicts where each
# tracked person SHOULD appear if only the camera moved. Then we match
# against those predicted positions instead of raw old positions.
#
# Result: IDs survive even large camera pans/rotations → no more
# everyone resetting to "Warming Up" every time the drone moves.
# -----------------------------------------------------------------------
class SimpleTracker:
    def __init__(self, max_dist=80):
        self.next_id  = 1
        self.tracks   = {}    # id -> (cx, cy)  raw image positions
        self.max_dist = max_dist

    def update(self, centroids, M=None):
        if not centroids:
            self.tracks = {}
            return {}

        if not self.tracks:
            result = {}
            for cx, cy in centroids:
                result[self.next_id] = (cx, cy)
                self.next_id += 1
            self.tracks = result
            return result

        # warp old positions using the camera transform
        # this predicts where each person should be after camera moved
        predicted = {}
        for tid, (tx, ty) in self.tracks.items():
            predicted[tid] = warp_point(M, tx, ty)

        used_ids = set()
        result   = {}

        for cx, cy in centroids:
            best_id   = None
            best_dist = self.max_dist

            # match against PREDICTED (camera-warped) positions, not raw old ones
            for tid, (px, py) in predicted.items():
                if tid in used_ids:
                    continue
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id   = tid

            if best_id is not None:
                result[best_id] = (cx, cy)
                used_ids.add(best_id)
            else:
                result[self.next_id] = (cx, cy)
                self.next_id += 1

        self.tracks = result
        return result


def draw_hud(frame, summary, frame_num, dx, dy):
    lines = [
        f"Frame     : {frame_num}",
        f"People    : {summary['total']}",
        f"Walking   : {summary['walking']}",
        f"Standing  : {summary['standing']}",
        f"Warming Up: {summary['warming_up']}",
        f"Cam shift : {dx:+.1f}px  {dy:+.1f}px",
    ]

    panel_x, panel_y = 10, 10
    line_h  = 22
    panel_w = 260
    panel_h = len(lines) * line_h + 16

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (panel_x + 8, panel_y + 16 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_HUD_TEXT, 1, cv2.LINE_AA)


def draw_person(frame, bbox, label_id, state, speed, centroid):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = centroid

    if state == "Walking":
        color = COLOR_WALKING
    elif state == "Standing":
        color = COLOR_STANDING
    else:
        color = COLOR_WARMUP

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 4, color, -1)

    label = f"ID:{label_id}  {state}  {speed}px/f"
    cv2.putText(frame, label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def run_pipeline(video_path,
                 model_path="best.pt",
                 output_path="output_processed.mp4",
                 buffer_size=15,
                 pixel_threshold=10.0):

    model      = YOLO(model_path)
    classifier = CrowdMovementClassifier(buffer_size=buffer_size,
                                          pixel_threshold=pixel_threshold)
    tracker    = SimpleTracker(max_dist=80)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    print(f"[INFO] Processing : {video_path}")
    print(f"[INFO] Resolution : {frame_w}x{frame_h}  |  FPS: {fps}")
    print(f"[INFO] Output     : {output_path}\n")

    frame_num = 0
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------------
        # Step 1 — estimate how the camera moved this frame
        # M handles pan + tilt + rotation + zoom all at once
        # -------------------------------------------------------------------
        M, cam_dx, cam_dy = estimate_camera_transform(prev_gray, curr_gray)
        classifier.update_camera_motion(cam_dx, cam_dy)
        prev_gray = curr_gray

        # -------------------------------------------------------------------
        # Step 2 — detect heads/people
        # -------------------------------------------------------------------
        results = model.predict(frame,
                                classes=[0],
                                imgsz=832,
                                conf=0.25,
                                iou=0.75,
                                max_det=500,
                                verbose=False)

        boxes_data   = results[0].boxes
        frame_states = []

        if boxes_data is not None and len(boxes_data) > 0:
            bboxes = boxes_data.xyxy.cpu().numpy()

            centroids = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroids.append((cx, cy))

            # pass M so tracker can warp old positions before matching
            # this is what keeps IDs stable across camera movements
            id_map = tracker.update(centroids, M=M)

            for bbox, (cx, cy) in zip(bboxes, centroids):
                tid = next((k for k, v in id_map.items() if v == (cx, cy)), None)
                if tid is None:
                    continue

                state, centroid, speed = classifier.classify(tid, bbox)
                frame_states.append(state)
                draw_person(frame, bbox, tid, state, speed, centroid)

        active_ids = list(tracker.tracks.keys())
        classifier.remove_lost_tracks(active_ids)

        summary = classifier.get_crowd_summary(frame_states)
        draw_hud(frame, summary, frame_num, cam_dx, cam_dy)
        writer.write(frame)

        if frame_num % 50 == 0:
            print(f"  Frame {frame_num:>5} | "
                  f"People: {summary['total']:>3} | "
                  f"Walking: {summary['walking']:>3} | "
                  f"Standing: {summary['standing']:>3} | "
                  f"WarmUp: {summary['warming_up']:>3} | "
                  f"Cam: ({cam_dx:+.1f}, {cam_dy:+.1f})")

    cap.release()
    writer.release()
    print(f"\n[INFO] Done! Output saved to: {output_path}")