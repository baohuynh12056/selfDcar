# src/env/game_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import time
from game_car_ai.src.vision.capture import ScreenCapture
from game_car_ai.src.detection.car_detector import CarDetector
from game_car_ai.src.controls.input import AndroidController
from game_car_ai.src.controls.keymap import GameKeyMap, DRIVING_ACTIONS
from game_car_ai.src.controls.new_input import ScrcpyController

class CarGameEnv(gym.Env):
    def __init__(self, max_opponents=4):
        super(CarGameEnv, self).__init__()
        
        # Initialize components
        self.render_mode = 'human'
        self.capture = ScreenCapture()
        self.capture.start()
        self.detector = CarDetector()
        self.controller2  = ScrcpyController()
        self.controller = AndroidController()
        self.keymap = GameKeyMap(self.controller2) 
        self.game_over = False       

        # Action space: [left, right, down, no-op]
        self.action_space = spaces.Discrete(len(DRIVING_ACTIONS))
        self.max_opponents = max_opponents        
        self.state_dim = 5 + (self.max_opponents * 5)

        # Observation space: preprocessed frame
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.screen_width = 640  
        self.screen_height = 640

        # Game state
        self.current_frame = None
        self.last_action = None
        self.episode_reward = 0
        self.steps = 0
        self.max_steps = 1000
        self.last_score = 0
        self.best_score = 0
    
    def step(self, action): 
        # 1. Execute action
        if action != self.last_action:
            self.keymap.execute(action)
            self.last_action = action
        time.sleep(0.1)
        observation = self._get_observation()
        
        # 2. Calculate reward
        done = self._check_game_over()
        reward = self._calculate_reward(observation,done)
        self.episode_reward += reward

        # 3. Check termination
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        info = {
            'steps': self.steps,
            'total_reward': self.episode_reward,
            'game_over': done,
            'score': self.last_score
        }
        
        return observation, reward, done, False, info
    
    def _extract_features(self, detections):
        """
        Convert YOLO detections to feature vector
        Format: [player_features, opp1_features, opp2_features, ...]
        """
        # Initialize feature vector with zeros
        features = np.zeros(self.state_dim, dtype=np.float32)
        
        player_features = []
        opponent_features = []
        
        # Classify detections (ignore menu_game)
        for det in detections:
            if det['class_name'] == 'player_car':
                # Normalize bbox coordinates [0, 1]
                x1, y1, x2, y2 = det['bbox']
                player_features.extend([
                    x1 / self.screen_width, 
                    y1 / self.screen_height,
                    (x2 - x1) / self.screen_width,  # width
                    (y2 - y1) / self.screen_height,  # height
                    det['confidence']  # confidence
                ])
            elif det['class_name'] == 'opponent_car':
                x1, y1, x2, y2 = det['bbox']
                opponent_features.extend([
                    x1 / self.screen_width, 
                    y1 / self.screen_height,
                    (x2 - x1) / self.screen_width,
                    (y2 - y1) / self.screen_height,
                    det['confidence']
                ])
        
        # Fill feature vector
        idx = 0
        
        # Player features (5 values)
        if player_features:
            features[idx:idx+5] = player_features[:5]
        idx += 5
        
        # Opponent features (up to 4 opponents)
        for i in range(self.max_opponents):
            if i < len(opponent_features) // 5:
                start_idx = i * 5
                features[idx:idx+5] = opponent_features[start_idx:start_idx+5]
            idx += 5
        
        return features
    
    def _get_observation(self):
        """Get YOLO vector observation"""
        # Get current frame
        self.current_frame = self.capture.get_frame()
        
        # Get detections from YOLO
        detections = self.detector.detect(self.current_frame) if self.current_frame is not None else []
        
        # Extract features from detections
        observation = self._extract_features(detections)
        
        return observation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.episode_reward = 0
        self.last_score = 0
        
        # Restart game
        self._restart_game()
        self.game_over = False

        # Get initial observation
        observation = self._get_observation()
        return observation, {}
    
    def _calculate_reward(self, observation, done):
        """
        Calculate reward based on survival and collisions.
        - done: True if game over
        - observation: current YOLO vector
        """
        if done:
            return -10.0  # Large penalty when game over

        # Base reward: survival
        reward = 0.1

        # Collision penalty
        collision_penalty = self._check_collisions(observation)  # returns 0 or -10

        reward += collision_penalty
        return reward

    
    def _check_game_over(self):
        """
        Check if game is over by detecting menu_game
        Returns: True if menu_game is detected (game over)
        """
        if self.current_frame is None:
            return False
        
        # Get detections from YOLO
        detections = self.detector.detect(self.current_frame)
        
        # Check if any detection is menu_game
        for det in detections:
            if det['class_name'] == 'menu_game':
                print("Game over detected: menu_game appeared")
                self.game_over = True
                return True
        
        return False

    def _restart_game(self):
        """Restart the game when game over"""
        print("🔄 Restarting game...")
        time.sleep(0.3) 
        # Only restart if game over was detected
        if self.game_over:
            self.controller.tap(1486, 944) 
            time.sleep(3)
        else:
            # If not game over, just reset state
            print("🔄 Resetting environment state...")
        
        self.game_over = False    

    def _line_intersects_bbox(self, line_p1, line_p2, bbox):
        """
        Check if bbox intersects with a line segment.
        bbox = [xmin, ymin, xmax, ymax]
        line_p1, line_p2 = (x,y)
        """
        xmin, ymin, xmax, ymax = bbox

        # Bbox corners
        corners = [
            (xmin, ymin),  # top-left
            (xmax, ymin),  # top-right
            (xmax, ymax),  # bottom-right
            (xmin, ymax)   # bottom-left
        ]

        # Helper: check if a point lies on a segment
        def point_on_segment(p, a, b, eps=1e-6):
            cross = (p[1]-a[1])*(b[0]-a[0]) - (p[0]-a[0])*(b[1]-a[1])
            if abs(cross) > eps:
                return False
            dot = (p[0]-a[0])*(b[0]-a[0]) + (p[1]-a[1])*(b[1]-a[1])
            if dot < 0:
                return False
            sq_len = (b[0]-a[0])**2 + (b[1]-a[1])**2
            if dot > sq_len:
                return False
            return True

        # 1. If any corner of bbox lies on the guardrail
        for c in corners:
            if point_on_segment(c, line_p1, line_p2):
                return True

        # 2. If guardrail intersects any edge of bbox
        edges = [
            ((xmin, ymin), (xmax, ymin)),  # top
            ((xmax, ymin), (xmax, ymax)),  # right
            ((xmax, ymax), (xmin, ymax)),  # bottom
            ((xmin, ymax), (xmin, ymin))   # left
        ]

        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        for edge in edges:
            if intersect(line_p1, line_p2, edge[0], edge[1]):
                return True

        return False


    def _check_collisions(self, observation):
        """
        Check collisions using YOLO features
        Returns: 0 if no collision, -10 if collision
        """
        player_x, player_y, player_w, player_h, player_conf = observation[0:5]
        
        # Player bounding box
        player_bbox = [player_x, player_y, player_x + player_w, player_y + player_h]

        # Check collision with each opponent
        for i in range(self.max_opponents):
            opp_idx = 5 + i * 5
            if np.any(observation[opp_idx:opp_idx+4] > 0):  # has opponent
                opp_x, opp_y, opp_w, opp_h, opp_conf = observation[opp_idx:opp_idx+5]
                opp_bbox = [opp_x, opp_y, opp_x + opp_w, opp_y + opp_h]
                
                if self._bbox_iou(player_bbox, opp_bbox) > 0.0:
                    return -10.0  # Collision penalty
       
        # Coordinates (normalized by image width = 640) of the guardrails
        # Left guardrail
        Left_guardrail_1 = (134/640.0, 389/640.0)
        Left_guardrail_2 = (84/640.0, 551/640.0)
        # Right guardrail
        Right_guardrail_1 = (513/640.0, 374/640.0)
        Right_guardrail_2 = (579/640.0, 549/640.0)

        if self._line_intersects_bbox(Left_guardrail_1, Left_guardrail_2, player_bbox) or self._line_intersects_bbox(Right_guardrail_1, Right_guardrail_2, player_bbox):
            return -5

        return 0.0  # No collision

    def _bbox_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two normalized bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0   
        
    def render(self, mode='human'):
        if self.current_frame is not None:
                detections = self.detector.detect(self.current_frame)
                frame_viz = self.detector.visualize(self.current_frame, detections)

                # Basic information
                cv2.putText(frame_viz, f"Steps: {self.steps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_viz, f"Reward: {self.episode_reward:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_viz, f"Game Over: {self.game_over}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display last action
                action_map = {0: "LEFT", 1: "RIGHT", 2: "DOWN", 3: "NOOP"}
                if hasattr(self, "last_action"):
                    action_name = action_map.get(self.last_action, "UNKNOWN")
                    cv2.putText(frame_viz, f"AI Action: {action_name}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Highlight menu_game detection
                menu_detections = [d for d in detections if d['class_name'] == 'menu_game']
                if menu_detections:
                    cv2.putText(frame_viz, "GAME OVER DETECTED!", (10, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                cv2.imshow('Car Game Environment', frame_viz)
                cv2.waitKey(1)

    def close(self):
        """Cleanup environment resources"""
        print("🧹 Cleaning up CarGameEnv...")
        
        if self.capture:
            self.capture.stop()
            self.capture = None
        
        if self.controller:
            try:
                self.controller.close()
            except AttributeError:
                pass
            self.controller = None
        
        cv2.destroyAllWindows()

def test_yolo_env():
    env = CarGameEnv()
    
    print("🧪 Testing YOLO vector environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    
    # Test some steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Obs={obs[:5]}")
        
        if done:
            obs, info = env.reset()
            print("Game over! Resetting...")
    
    env.close()

if __name__ == "__main__":
    test_yolo_env()
