import math
from collections import deque, defaultdict


class CrowdMovementClassifier:
    """
    Classifies each person as Walking, Standing, or Warming Up.

    Camera motion compensation:
      The drone can pan, tilt, rotate, zoom. We receive the camera's
      translation (dx, dy) each frame and accumulate it over the buffer
      window. A person's REAL movement = raw movement - camera movement.
    """

    def __init__(self, buffer_size=15, pixel_threshold=10.0):
        self.buffer_size     = buffer_size
        self.pixel_threshold = pixel_threshold
        self.track_history   = defaultdict(lambda: deque(maxlen=buffer_size))

        # stores (cam_dx, cam_dy) per frame over the buffer window
        self.cam_buffer = deque(maxlen=buffer_size)

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update_camera_motion(self, cam_dx, cam_dy):
        """Call once per frame with camera translation from affine matrix."""
        self.cam_buffer.append((cam_dx, cam_dy))

    def classify(self, track_id, bbox):
        cx, cy = self.get_centroid(bbox)
        self.track_history[track_id].append((cx, cy))
        history = self.track_history[track_id]

        if len(history) < self.buffer_size:
            return "Warming Up", (cx, cy), 0.0

        oldest = history[0]
        newest = history[-1]

        # how much the person's centroid moved in image space
        raw_dx = newest[0] - oldest[0]
        raw_dy = newest[1] - oldest[1]

        # how much the camera translated over the same window
        cum_cam_dx = sum(d[0] for d in self.cam_buffer)
        cum_cam_dy = sum(d[1] for d in self.cam_buffer)

        # subtract camera motion → real person movement
        real_dx = raw_dx - cum_cam_dx
        real_dy = raw_dy - cum_cam_dy

        displacement = math.sqrt(real_dx**2 + real_dy**2)
        speed        = displacement / self.buffer_size

        state = "Walking" if displacement >= self.pixel_threshold else "Standing"
        return state, (cx, cy), round(speed, 1)

    def get_crowd_summary(self, current_states):
        return {
            "total"     : len(current_states),
            "walking"   : current_states.count("Walking"),
            "standing"  : current_states.count("Standing"),
            "warming_up": current_states.count("Warming Up"),
        }

    def remove_lost_tracks(self, active_ids):
        for tid in list(self.track_history.keys()):
            if tid not in active_ids:
                del self.track_history[tid]