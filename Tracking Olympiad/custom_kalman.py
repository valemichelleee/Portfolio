import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import math
from ultralytics import YOLO
import cv2
import pandas as pd

def _compute_histogram(frame, bbox):
    """Computes a color histogram for the given bounding box in the frame."""
    x, y, w, h = [int(v) for v in bbox]
    
    # Ensure the bounding box is within the frame boundaries
    h_frame, w_frame = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)

    if w <= 0 or h <= 0:
        return None

    obj_img = frame[y:y+h, x:x+w]
    hsv_obj = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)
    
    # Use H and S channels for histogram. Fewer bins for robustness.
    hist = cv2.calcHist([hsv_obj], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

class Track:
    """Represents a single object track with persistent object assumption"""
    def __init__(self, track_id, detection, frame_id, frame_size, frame=None):
        self.track_id = track_id
        self.detections = [detection]  # History of detections
        self.frame_ids = [frame_id]    # Frame IDs for each detection
        self.state = 'active'          # 'active', 'occluded', 'confirmed'
        self.age = 0                   # Frames since creation
        self.time_since_update = 0     # Frames since last update
        self.max_occlusion_time = 101   # Max frames of occlusion before marking as long-term occluded
        self.n_init = 3                # Min detections before track is confirmed
        self.hits = 1                  # Number of detection matches
        self.hit_streak = 1            # Consecutive detection matches
        self.occlusion_streak = 0      # Consecutive frames without detection
        self.frame_size = frame_size
        w, h = frame_size
        self.frame_diag = math.hypot(w, h)
        
        # Store velocity for better prediction during occlusion
        self.velocity_history = []
        self.max_velocity_history = 10
        self.histogram = None
        self.hist_alpha = 0.5
        
        # Kalman filter for motion prediction
        self.kf = self._init_kalman_filter(detection)
        
        # Initialize histogram
        if frame is not None:
            self.histogram = _compute_histogram(frame, detection['bbox'])

        # Confidence decay during occlusion
        self.base_confidence = detection['confidence']
        self.confidence_decay_rate = 0.95
        
    def _init_kalman_filter(self, detection):
        """Initialize Kalman filter for motion prediction"""
        # State: [x, y, w, h, vx, vy, vw, vh] (position, size, velocities)
        kf = cv2.KalmanFilter(8, 4)
        
        # Transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (observe position and size)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (increased for low frame rate)
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        kf.processNoiseCov[4:, 4:] *= 0.5
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance matrix
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        # Initial state
        x, y, w, h = detection['bbox']
        kf.statePre = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        return kf
    
    def predict(self):
        """Predict next state using Kalman filter"""
        self.kf.predict()
        predicted_state = self.kf.statePre.copy()
        
        # Calculate confidence based on occlusion time
        if self.time_since_update == 0:
            confidence = self.base_confidence
        else:
            # Decay confidence during occlusion
            confidence = self.base_confidence * (self.confidence_decay_rate ** self.time_since_update)
            confidence = max(confidence, 0.1)  # Minimum confidence
        
        return {
            'bbox': predicted_state[:4].copy(),
            'confidence': confidence,
            'class': self.detections[-1].get('class', 0)
        }
    
    def update(self, detection, frame_id, frame=None):
        """Update track with new detection"""
        # Update histogram with a moving average
        if frame is not None:
            new_hist = _compute_histogram(frame, detection['bbox'])
            if new_hist is not None:
                if self.histogram is None:
                    self.histogram = new_hist
                else:
                    self.histogram = cv2.addWeighted(self.histogram, self.hist_alpha, new_hist, 1 - self.hist_alpha, 0)

        # Calculate velocity if we have previous detections
        if len(self.detections) > 0:
            prev_detection = self.detections[-1]
            prev_frame = self.frame_ids[-1]
            
            if frame_id > prev_frame:
                dt = max(1, frame_id - prev_frame)
                
                # Calculate velocity
                dx = detection['bbox'][0] - prev_detection['bbox'][0]
                dy = detection['bbox'][1] - prev_detection['bbox'][1]
                dw = detection['bbox'][2] - prev_detection['bbox'][2]
                dh = detection['bbox'][3] - prev_detection['bbox'][3]
                
                damping = min(1.0, 3.0 / dt)
                velocity = [dx/dt * damping, dy/dt * damping, dw/dt * damping, dh/dt * damping]

                max_velocity = self.frame_diag * 0.3  # 30% of frame diagonal per frame
                velocity = [max(-max_velocity, min(max_velocity, v)) for v in velocity]

                self.velocity_history.append(velocity)
                
                # Keep only recent velocity history
                if len(self.velocity_history) > self.max_velocity_history:
                    self.velocity_history.pop(0)
        
        self.detections.append(detection)
        self.frame_ids.append(frame_id)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.occlusion_streak = 0
        self.state = 'active'
        self.base_confidence = detection['confidence']
        
        # Update Kalman filter
        x, y, w, h = detection['bbox']
        measurement = np.array([x, y, w, h], dtype=np.float32)
        self.kf.correct(measurement)
        
        if self.hits >= self.n_init:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed in current frame (temporary occlusion)"""
        self.hit_streak = 0
        self.time_since_update += 1
        self.occlusion_streak += 1
        
        # Instead of deleting, mark as occluded
        if self.time_since_update > self.max_occlusion_time:
            self.state = 'long_term_occluded'
        else:
            self.state = 'occluded'
    
    def get_predicted_position(self, future_frames=1):
        """Get predicted position for future frames using velocity"""
        if not self.velocity_history or self.time_since_update > 5:
            return self.predict()
        
        # Use average velocity for prediction
        avg_velocity = np.mean(self.velocity_history[-3:], axis=0)  # Use recent velocities
        
        # Get current prediction
        current_pred = self.predict()
        current_bbox = current_pred['bbox']
        
        # Apply velocity for future frames
        decay_factor = 0.8 ** self.time_since_update
        future_bbox = current_bbox.copy()
        future_bbox[0] += avg_velocity[0] * future_frames * decay_factor
        future_bbox[1] += avg_velocity[1] * future_frames * decay_factor
        # future_bbox[2] += avg_velocity[2] * future_frames
        # future_bbox[3] += avg_velocity[3] * future_frames
        future_bbox[0] = max(0, min(self.frame_size[0] - future_bbox[2], future_bbox[0]))
        future_bbox[1] = max(0, min(self.frame_size[1] - future_bbox[3], future_bbox[1]))
        
        return {
            'bbox': future_bbox,
            'confidence': current_pred['confidence'],
            'class': current_pred['class']
        }
    
    def to_dict(self):
        """Convert track to dictionary for output"""
        if not self.detections:
            return None
        
        # Always return track info, even if occluded
        if self.time_since_update == 0:
            # Use actual detection
            latest_detection = self.detections[-1]
            bbox = latest_detection['bbox']
            confidence = latest_detection['confidence']
        else:
            # Use prediction
            predicted = self.predict()
            bbox = predicted['bbox']
            confidence = predicted['confidence']
        
        return {
            'track_id': self.track_id,
            'bbox': bbox,
            'confidence': confidence,
            'class': self.detections[-1].get('class', 0),
            'state': self.state,
            'time_since_update': self.time_since_update,
            'is_occluded': self.time_since_update > 0
        }

class PersistentMultiObjectTracker:
    """Multi-object tracker assuming objects never leave the scene"""
    def __init__(self, frame_size, max_distance=0.15, iou_threshold=0.1, expected_objects=None):
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0
        self.iou_threshold = iou_threshold
        self.expected_objects = expected_objects
        w, h = frame_size
        self.frame_size = frame_size
        self.frame_diag = math.hypot(w, h)
        self.max_distance = max_distance * self.frame_diag
        
        # For handling reappearing objects
        self.association_threshold = 0.3  # Lower threshold for occluded objects
        
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to corner coordinates
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_distance(self, box1, box2):
        """Compute Euclidean distance between box centers"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _compute_cost_matrix(self, tracks, detections, frame):
        """Compute cost matrix for Hungarian algorithm"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        # Pre-compute histograms for all new detections
        det_histograms = [_compute_histogram(frame, det['bbox']) for det in detections]

        for i, track in enumerate(tracks):
            # Use different prediction strategies based on track state
            if track.state == 'occluded' and track.time_since_update > 1:
                predicted_detection = track.get_predicted_position()
            else:
                predicted_detection = track.predict()
            
            for j, detection in enumerate(detections):
                # Compute IoU and distance
                iou = self._compute_iou(predicted_detection['bbox'], detection['bbox'])
                distance = self._compute_distance(predicted_detection['bbox'], detection['bbox'])
                
                # Normalize distance (assuming reasonable image size)
                normalized_distance = distance / self.frame_diag
                
                # Adaptive thresholds based on track state
                if track.state == 'occluded':
                    # More lenient for occluded tracks
                    distance_threshold = self.max_distance * (1.5 + 0.1 * track.time_since_update)
                    iou_threshold = max(0.05, self.iou_threshold * 0.5)
                    
                    # Penalize based on occlusion time
                    occlusion_penalty = min(0.3, 0.02 * track.time_since_update)
                else:
                    distance_threshold = self.max_distance
                    iou_threshold = self.iou_threshold
                    occlusion_penalty = 0
                
                # Cost calculation
                if distance > distance_threshold and iou < iou_threshold:
                    cost = 1e6  # Very high cost for impossible matches
                else:
                    # Weighted combination of IoU and distance for motion cost
                    motion_cost = (1 - iou) * 0.4 + normalized_distance * 0.6 + occlusion_penalty
                    
                    # Appearance cost based on histogram comparison
                    appearance_cost = 0.5  # Default value if histogram is not available
                    det_hist = det_histograms[j]
                    if track.histogram is not None and det_hist is not None:
                        # Use Bhattacharyya distance for histogram comparison
                        appearance_cost = cv2.compareHist(track.histogram, det_hist, cv2.HISTCMP_BHATTACHARYYA)
                    
                    # Final cost is a weighted sum of motion and appearance costs
                    cost = 0.7 * motion_cost + 0.3 * appearance_cost

                    if track.hits > 5 and track.time_since_update < 3:
                        cost *= 0.9

                    confidence_weight = detection['confidence']
                    cost *= (2.0 - confidence_weight)

                    # Bonus for tracks that have been occluded (encourage reappearance)
                    if track.state == 'occluded':
                        cost *= 0.8  
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def update(self, detections, frame):
        """Update tracks with new detections"""
        self.frame_id += 1
        
        # Convert detections to required format if needed
        formatted_detections = []
        for det in detections:
            if isinstance(det, dict):
                formatted_detections.append(det)
            else:
                # Assume det is [x, y, w, h, confidence, class]
                formatted_detections.append({
                    'bbox': det[:4],
                    'confidence': det[4] if len(det) > 4 else 1.0,
                    'class': int(det[5]) if len(det) > 5 else 0
                })
        
        # All tracks are considered (no deletion)
        active_tracks = self.tracks
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(active_tracks, formatted_detections, frame)
        
        # Hungarian algorithm for optimal assignment
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter assignments based on adaptive thresholds
            valid_assignments = []
            for row, col in zip(row_indices, col_indices):
                track = active_tracks[row]
                cost = cost_matrix[row, col]
                
                # Adaptive threshold based on track state
                if track.state == 'occluded':
                    threshold = 1.0  
                else:
                    threshold = 1.0
                
                if cost < threshold:
                    valid_assignments.append((row, col))
            
            # Update matched tracks
            matched_tracks = set()
            matched_detections = set()
            
            for row, col in valid_assignments:
                active_tracks[row].update(formatted_detections[col], self.frame_id, frame)
                matched_tracks.add(row)
                matched_detections.add(col)
            
            # Mark unmatched tracks as occluded (not deleted)
            for i, track in enumerate(active_tracks):
                if i not in matched_tracks:
                    track.mark_missed()
            
            # Handle unmatched detections
            unmatched_detections = [
                formatted_detections[i] for i in range(len(formatted_detections)) 
                if i not in matched_detections
            ]
            
            # Only create new tracks if we haven't reached expected object count
            if self.expected_objects is None or len(self.tracks) < self.expected_objects:
                for detection in unmatched_detections:
                    new_track = Track(self.next_id, detection, self.frame_id, (self.frame_size[0], self.frame_size[1]), frame=frame)
                    self.tracks.append(new_track)
                    self.next_id += 1
            else:
                # If we have expected number of objects, try to associate with long-term occluded tracks
                for detection in unmatched_detections:
                    # Find the best long-term occluded track to reactivate
                    best_track = None
                    best_cost = float('inf')
                    
                    for track in self.tracks:
                        if track.state == 'long_term_occluded':
                            predicted = track.get_predicted_position()
                            cost = self._compute_distance(predicted['bbox'], detection['bbox'])
                            if cost < best_cost:
                                best_cost = cost
                                best_track = track
                    
                    if best_track and best_cost < self.max_distance * 2:
                        best_track.update(detection, self.frame_id, frame)
        
        else:
            # No existing tracks, create new ones
            for detection in formatted_detections:
                new_track = Track(self.next_id, detection, self.frame_id, self.frame_size, frame=frame)
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Age all tracks
        for track in self.tracks:
            track.age += 1
        
        return self.get_current_tracks()
    
    def get_current_tracks(self):
        """Get current tracks (including occluded ones)"""
        current_tracks = []
        for track in self.tracks:
            track_dict = track.to_dict()
            if track_dict:
                current_tracks.append(track_dict)
        return current_tracks
    
    def get_active_tracks(self):
        """Get only currently detected (non-occluded) tracks"""
        active_tracks = []
        for track in self.tracks:
            if track.state == 'active' or track.state == 'confirmed':
                track_dict = track.to_dict()
                if track_dict:
                    active_tracks.append(track_dict)
        return active_tracks
    
    def get_track_statistics(self):
        """Get statistics about tracks"""
        stats = {
            'total_tracks': len(self.tracks),
            'active_tracks': len([t for t in self.tracks if t.state in ['active', 'confirmed']]),
            'occluded_tracks': len([t for t in self.tracks if t.state == 'occluded']),
            'long_term_occluded': len([t for t in self.tracks if t.state == 'long_term_occluded'])
        }
        return stats
    
    def get_track_history(self, track_id):
        """Get history of a specific track"""
        for track in self.tracks:
            if track.track_id == track_id:
                return {
                    'track_id': track_id,
                    'detections': track.detections,
                    'frame_ids': track.frame_ids,
                    'state': track.state,
                    'time_since_update': track.time_since_update
                }
        return None

def main():
    # Load YOLOv8 model
    model = YOLO("runs/train/hexbug_v8x_1/weights/best.pt") 
    video_in_path  = 'test021.mp4'
    object_number = 4
    video_out_path = "test021_tracked.mp4"
    csv_out_path   = "test021.csv"

    # Open video file
    cap = cv2.VideoCapture(video_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width,height))

    rows = []

    # Create tracker
    tracker = PersistentMultiObjectTracker(
        frame_size=(width, height),
        max_distance=0.15,
        iou_threshold=0.1,
        expected_objects=object_number
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO inference
        results = model(frame)[0]

        # Extract detections as [x, y, w, h, confidence]
        dets = []
        # YOLOv8 .boxes.xyxy gives (x1,y1,x2,y2)
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(),
                            results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            dets.append([x1, y1, w, h, float(conf)])

        # Update tracker
        tracks = tracker.update(dets, frame)

        # Visualize
        for tr in tracks:
            tid = tr["track_id"]
            x, y, w, h = tr['bbox']
            # center coordinates
            xc = x + w/2
            yc = y + h/2

            rows.append({
            "t":       frame_idx-1,
            "hexbug":  tid-1,
            "x":       xc,
            "y":       yc
            })

            col = (0,255,0) if not tr['is_occluded'] else (0,128,255)
            cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), col, 2)
            cv2.putText(frame, f"ID {tid}", (int(x),int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        writer.write(frame)

    cap.release()
    df = pd.DataFrame(rows, columns=["t","hexbug","x","y"])
    df.to_csv(csv_out_path)
    print("Done! CSV →", csv_out_path, "   Video →", video_out_path)


if __name__ == "__main__":
    main()