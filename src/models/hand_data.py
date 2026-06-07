from src.models.sim_normalized_landmarks import SimNormalizedLandmark

"""
O. WRIST
11. MIDDLE_FINGER_DIP
1. THUMB_CMC
12. MIDDLE_FINGER_TIP
2. THUMB_MCP
13. RING_FINGER_MCP
3. THUMB_IP
14. RING_FINGER_PIP
4. THUMB_TIP
15. RING_FINGER_DIP
5. INDEX_FINGER_MCP
16. RING_FINGER_TIP
6. INDEX_FINGER_PIP
17. PINKY_MCP
7. INDEX_FINGER_DIP
18. PINKY_PIP
8. INDEX_FINGER_TIP
19. PINKY-DIP
9. MIDDLE_FINGER_MCP
20. PINKY_TIP
10. MIDDLE_FINGER_PIP
"""

FINGER_TIPS = [4, 8, 12, 16, 20]


class HandData:
    def __init__(
        self,
        handedness,
        gesture,
        score,
        hand_landmarks: list[SimNormalizedLandmark],
        hand_world_landmarks,
    ):
        self.handedness = handedness
        self.gesture = gesture
        self.score = score
        self.hand_landmarks = hand_landmarks
        self.hand_world_landmarks = hand_world_landmarks
        self.hand_index = self._hand_index(handedness)

    def _hand_index(self, handedness):
        if handedness[0].category_name == "Right":
            return 1
        else:
            return 0

    def get_index_finger_landmark(self):
        return self.hand_landmarks[8]

    def get_thumb_finger_landmark(self):
        return self.hand_landmarks[4]

    def get_hand_stats(self):
        return {
            "handedness": self.handedness[0].category_name,
            "gesture": self.gesture,
            "score": self.score,
            "hand_index": self.hand_index,
        }

    def get_finger_tips(self):
        finger_tips = []
        for idx in FINGER_TIPS:
            finger_tips.append(self.hand_landmarks[idx])
        return finger_tips

    def get_pinch_fingertips(self):
        return [self.hand_landmarks[4], self.hand_landmarks[8]]
