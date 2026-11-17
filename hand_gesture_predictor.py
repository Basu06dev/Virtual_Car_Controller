import time
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# PATH SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand"
    / "hand_landmarker.task"
)


# ==================================================
# CAMERA SETTINGS
# ==================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ==================================================
# HAND DETECTION SETTINGS
# ==================================================

NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_HAND_TRACKING_CONFIDENCE = 0.6


# ==================================================
# GESTURE SETTINGS
# ==================================================

# Difference in normalized Y coordinate required
# to consider one hand higher than the other.

HAND_HEIGHT_THRESHOLD = 0.12

# Both hands must be approximately level.

LEVEL_THRESHOLD = 0.12

# Stop requires both fists to be raised
# above the calibrated neutral position.

STOP_RAISE_THRESHOLD = 0.12

# Minimum number of extended fingers
# required to classify a hand as open.

OPEN_FINGER_THRESHOLD = 3


# ==================================================
# DRAW TEXT
# ==================================================

def draw_text(
    frame,
    text,
    position,
    color=(0, 255, 0),
    scale=0.7,
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


# ==================================================
# DRAW HAND LANDMARKS
# ==================================================

def draw_hand_landmarks(frame, landmarks):

    height, width, _ = frame.shape

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17),
    ]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )

    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2,
        )  


# ==================================================
# HAND FEATURE FUNCTIONS
# ==================================================

def calculate_palm_center(landmarks):

    palm_indices = [
        0,
        5,
        9,
        13,
        17,
    ]

    x_values = [
        landmarks[index].x
        for index in palm_indices
    ]

    y_values = [
        landmarks[index].y
        for index in palm_indices
    ]

    center_x = sum(x_values) / len(x_values)
    center_y = sum(y_values) / len(y_values)

    return center_x, center_y


def is_finger_extended(
    landmarks,
    tip_index,
    pip_index,
):

    tip_y = landmarks[tip_index].y
    pip_y = landmarks[pip_index].y

    return tip_y < pip_y


def count_extended_fingers(landmarks):

    fingers = [
        (8, 6),     # Index
        (12, 10),   # Middle
        (16, 14),   # Ring
        (20, 18),   # Little
    ]

    extended_count = 0

    for tip_index, pip_index in fingers:

        if is_finger_extended(
            landmarks,
            tip_index,
            pip_index,
        ):

            extended_count += 1

    return extended_count


def is_open_hand(landmarks):

    extended_count = count_extended_fingers(
        landmarks
    )

    return (
        extended_count >= OPEN_FINGER_THRESHOLD
    )


def is_closed_hand(landmarks):

    extended_count = count_extended_fingers(
        landmarks
    )

    return (
        extended_count <= 1
    )


# ==================================================
# HAND INFORMATION
# ==================================================

def extract_hand_information(
    result,
    frame,
):

    hands = {}

    if not result.hand_landmarks:

        return hands

    for index, landmarks in enumerate(
        result.hand_landmarks
    ):

        if index >= len(result.handedness):

            continue

        handedness_list = (
            result.handedness[index]
        )

        if not handedness_list:

            continue

        handedness = (
            handedness_list[0].category_name
        )

        center_x, center_y = (
            calculate_palm_center(
                landmarks
            )
        )

        extended_fingers = (
            count_extended_fingers(
                landmarks
            )
        )

        open_hand = is_open_hand(
            landmarks
        )

        closed_hand = is_closed_hand(
            landmarks
        )

        hands[handedness] = {
            "landmarks": landmarks,
            "center_x": center_x,
            "center_y": center_y,
            "extended_fingers": extended_fingers,
            "open": open_hand,
            "closed": closed_hand,
        }

        draw_hand_landmarks(
            frame,
            landmarks,
        )

    return hands


# ==================================================
# GESTURE CLASSIFIER
# ==================================================

class GestureClassifier:

    def __init__(self):

        self.neutral_center_y = None

    def reset_calibration(self):

        self.neutral_center_y = None

    def calibrate(self, hands):

        if (
            "Left" not in hands
            or "Right" not in hands
        ):

            return False

        left_hand = hands["Left"]
        right_hand = hands["Right"]

        # Both hands must be fists.

        if not (
            left_hand["closed"]
            and right_hand["closed"]
        ):

            return False

        left_y = left_hand["center_y"]
        right_y = right_hand["center_y"]

        height_difference = abs(
            left_y - right_y
        )

        # Both fists must be approximately level.

        if height_difference > LEVEL_THRESHOLD:

            return False

        self.neutral_center_y = (
            left_y + right_y
        ) / 2.0

        return True

    def classify(self, hands):

        if (
            "Left" not in hands
            or "Right" not in hands
        ):

            return "NO_HANDS"

        left_hand = hands["Left"]
        right_hand = hands["Right"]

        left_y = left_hand["center_y"]
        right_y = right_hand["center_y"]

        average_y = (
            left_y + right_y
        ) / 2.0

        height_difference = abs(
            left_y - right_y
        )

        hands_are_level = (
            height_difference <= LEVEL_THRESHOLD
        )

        both_closed = (
            left_hand["closed"]
            and right_hand["closed"]
        )

        both_open = (
            left_hand["open"]
            and right_hand["open"]
        )

        # One hand open + one hand closed = STOP

        one_open_one_closed = (
            (
                left_hand["open"]
                and right_hand["closed"]
            )
            or
            (
                left_hand["closed"]
                and right_hand["open"]
            )
)
        # ------------------------------------------
        # STOP
        # ------------------------------------------

        # One hand open + one hand closed.

        if one_open_one_closed:

            return "STOP"
        # ==========================================
        # FORWARD
        # ==========================================

        # Both hands open and approximately level.

        if (
            both_open
            and hands_are_level
        ):

            return "NEUTRAL"

        # ==========================================
        # CLOSED-HAND GESTURES
        # ==========================================

        if both_closed:

            # ======================================
            # IMPORTANT STEERING LOGIC
            # ======================================

            # MediaPipe hand labels are being used
            # with the mirrored camera image.
            #
            # YOUR REQUIRED MAPPING:
            #
            # Physical RIGHT hand raised
            #                ↓
            #             LEFT turn
            #
            # Physical LEFT hand raised
            #                ↓
            #            RIGHT turn
            #
            # Therefore the output directions are
            # intentionally swapped here.

            # --------------------------------------
            # RIGHT HAND RAISED → LEFT
            # --------------------------------------

            if (
                right_y
                < left_y - HAND_HEIGHT_THRESHOLD
            ):

                return "RIGHT"

            # --------------------------------------
            # LEFT HAND RAISED → RIGHT
            # --------------------------------------

            if (
                left_y
                < right_y - HAND_HEIGHT_THRESHOLD
            ):

                return "LEFT"

            # ======================================
            # NEUTRAL
            # ======================================

            if hands_are_level:

                return "FORWARD"

        # ==========================================
        # UNKNOWN
        # ==========================================

        return "UNKNOWN"


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    print(
        "Starting Hand Gesture Controller..."
    )

    print()

    # ==========================================
    # CHECK MODEL
    # ==========================================

    if not HAND_MODEL_PATH.exists():

        print(
            "ERROR: Hand model not found:"
        )

        print(HAND_MODEL_PATH)

        return

    # ==========================================
    # CREATE HAND LANDMARKER
    # ==========================================

    base_options = python.BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH
        )
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=(
            MIN_HAND_DETECTION_CONFIDENCE
        ),
        min_hand_presence_confidence=(
            MIN_HAND_PRESENCE_CONFIDENCE
        ),
        min_tracking_confidence=(
            MIN_HAND_TRACKING_CONFIDENCE
        ),
    )

    landmarker = (
        vision.HandLandmarker.create_from_options(
            options
        )
    )

    # ==========================================
    # CAMERA
    # ==========================================

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        landmarker.close()

        return

    # ==========================================
    # VARIABLES
    # ==========================================

    classifier = GestureClassifier()

    calibration_active = False

    calibration_frames = 0

    calibration_target = 60

    last_timestamp_ms = 0

    previous_time = time.perf_counter()

    current_gesture = "WAITING"

    # ==========================================
    # MAIN LOOP
    # ==========================================

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read frame."
                )

                break

            # Mirror camera.

            frame = cv2.flip(
                frame,
                1
            )

            # ==================================
            # FPS
            # ==================================

            current_time = time.perf_counter()

            elapsed = (
                current_time
                - previous_time
            )

            previous_time = current_time

            fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            # ==================================
            # TIMESTAMP
            # ==================================

            timestamp_ms = int(
                current_time * 1000
            )

            if (
                timestamp_ms
                <= last_timestamp_ms
            ):

                timestamp_ms = (
                    last_timestamp_ms + 1
                )

            last_timestamp_ms = timestamp_ms

            # ==================================
            # MEDIAPIPE IMAGE
            # ==================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # ==================================
            # GET HANDS
            # ==================================

            hands = extract_hand_information(
                result,
                frame,
            )

            # ==================================
            # CALIBRATION
            # ==================================

            if calibration_active:

                if classifier.calibrate(hands):

                    calibration_frames += 1

                    draw_text(
                        frame,
                        (
                            f"Calibrating: "
                            f"{calibration_frames}/"
                            f"{calibration_target}"
                        ),
                        (20, 40),
                        (0, 255, 255),
                    )

                    if (
                        calibration_frames
                        >= calibration_target
                    ):

                        calibration_active = False

                        print(
                            "Neutral calibration completed."
                        )

                        current_gesture = (
                            "NEUTRAL"
                        )

                else:

                    draw_text(
                        frame,
                        "Hold both closed fists level",
                        (20, 40),
                        (0, 255, 255),
                    )

            # ==================================
            # CLASSIFICATION
            # ==================================

            if not calibration_active:

                current_gesture = (
                    classifier.classify(
                        hands
                    )
                )

            # ==================================
            # GESTURE COLOR
            # ==================================

            if current_gesture in [
                "NEUTRAL",
                "FORWARD",
                "LEFT",
                "RIGHT",
                "STOP",
            ]:

                gesture_color = (
                    0,
                    255,
                    0,
                )

            else:

                gesture_color = (
                    0,
                    0,
                    255,
                )

            # ==================================
            # DISPLAY GESTURE
            # ==================================

            draw_text(
                frame,
                f"Gesture: {current_gesture}",
                (20, 90),
                gesture_color,
                0.9,
            )

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                (20, 130),
            )

            draw_text(
                frame,
                "C: Calibrate neutral",
                (20, 180),
                (255, 255, 0),
                0.6,
            )

            draw_text(
                frame,
                "B: Reset calibration",
                (20, 210),
                (255, 255, 0),
                0.6,
            )

            draw_text(
                frame,
                "Q/ESC: Exit",
                (20, 240),
                (255, 255, 0),
                0.6,
            )

            # ==================================
            # LEFT HAND STATUS
            # ==================================

            if "Left" in hands:

                left_status = (
                    "OPEN"
                    if hands["Left"]["open"]
                    else "CLOSED"
                    if hands["Left"]["closed"]
                    else "UNKNOWN"
                )

                draw_text(
                    frame,
                    f"Left: {left_status}",
                    (20, 290),
                    (255, 255, 255),
                    0.6,
                )

            # ==================================
            # RIGHT HAND STATUS
            # ==================================

            if "Right" in hands:

                right_status = (
                    "OPEN"
                    if hands["Right"]["open"]
                    else "CLOSED"
                    if hands["Right"]["closed"]
                    else "UNKNOWN"
                )

                draw_text(
                    frame,
                    f"Right: {right_status}",
                    (20, 320),
                    (255, 255, 255),
                    0.6,
                )

            # ==================================
            # SHOW CAMERA
            # ==================================

            cv2.imshow(
                "Hand Gesture Virtual Car Controller",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            # ==================================
            # CALIBRATE
            # ==================================

            if key == ord("c"):

                classifier.reset_calibration()

                calibration_active = True

                calibration_frames = 0

                current_gesture = (
                    "CALIBRATING"
                )

                print()

                print(
                    "Calibration started."
                )

                print(
                    "Keep both fists level "
                    "in the neutral position."
                )

            # ==================================
            # RESET
            # ==================================

            elif key == ord("b"):

                classifier.reset_calibration()

                calibration_active = False

                calibration_frames = 0

                current_gesture = (
                    "WAITING"
                )

                print(
                    "Calibration reset."
                )

            # ==================================
            # EXIT
            # ==================================

            elif (
                key == ord("q")
                or key == 27
            ):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        landmarker.close()

        print(
            "Hand gesture controller stopped."
        )


# ==================================================
# PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()