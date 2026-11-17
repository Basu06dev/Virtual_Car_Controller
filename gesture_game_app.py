
"""
Universal AI Gesture Game Controller
------------------------------------

This application:
1. Detects hand gestures using MediaPipe.
2. Converts gestures into game commands.
3. Sends keyboard input through GameController.
4. Supports different game profiles.
5. Includes safety controls and cleanup.

Controls:
    G       Enable / disable game control
    Q/ESC   Exit application

Available profiles:
    default -> Arrow keys
    wasd    -> W, A, S, D
    racing  -> Arrow keys
"""

import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_gesture_predictor import (
    HAND_MODEL_PATH,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_HAND_TRACKING_CONFIDENCE,
    extract_hand_information,
    GestureClassifier,
    draw_text,
)

from game_controller import GameController


# ============================================================
# APPLICATION SETTINGS
# ============================================================

WINDOW_NAME = "AI Gesture Racing Controller"

# Camera window size
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 320

# Camera window position
WINDOW_X = 20
WINDOW_Y = 50

# Keep the camera window above other windows
WINDOW_ALWAYS_ON_TOP = True

# Game control starts disabled for safety
CONTROL_ENABLED = False

# ============================================================
# SELECT GAME PROFILE
# ============================================================

# Available profiles:
#
# "default" -> Arrow keys
# "wasd"    -> W, A, S, D
# "racing"  -> Arrow keys

GAME_PROFILE = "default"


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    global CONTROL_ENABLED

    print("Starting AI Gesture Racing Controller...")
    print()

    print("Controls:")
    print("G       Enable / disable game control")
    print("Q/ESC   Exit application")
    print()

    print(f"Selected game profile: {GAME_PROFILE}")
    print()

    # ========================================================
    # CHECK HAND MODEL
    # ========================================================

    if not HAND_MODEL_PATH.exists():

        print("ERROR: Hand model not found:")
        print(HAND_MODEL_PATH)

        return

    # ========================================================
    # CREATE HAND LANDMARKER
    # ========================================================

    base_options = python.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
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

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        landmarker.close()

        return

    # ========================================================
    # CREATE OBJECTS
    # ========================================================

    classifier = GestureClassifier()

    # Select the configured game profile
    controller = GameController(
        profile=GAME_PROFILE
    )

    # Timestamp required by MediaPipe VIDEO mode
    last_timestamp_ms = 0

    # FPS tracking
    previous_time = time.perf_counter()

    # Current detected gesture
    current_gesture = "WAITING"

    # Track previous control state
    previous_control_enabled = False

    # ========================================================
    # CREATE CAMERA WINDOW
    # ========================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    cv2.moveWindow(
        WINDOW_NAME,
        WINDOW_X,
        WINDOW_Y
    )

    # Keep window above other applications
    if WINDOW_ALWAYS_ON_TOP:

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_TOPMOST,
            1
        )

    print("Floating camera window created.")

    print(
        f"Window size: "
        f"{WINDOW_WIDTH} x {WINDOW_HEIGHT}"
    )

    print()
    print("Press G inside the camera window to enable control.")
    print("Press G again to disable control.")
    print("Press Q or ESC to exit.")
    print()

    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    try:

        while True:

            # =================================================
            # READ CAMERA FRAME
            # =================================================

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read camera frame."
                )

                break

            # =================================================
            # MIRROR CAMERA
            # =================================================

            frame = cv2.flip(frame, 1)

            # =================================================
            # CALCULATE FPS
            # =================================================

            current_time = time.perf_counter()

            elapsed = (
                current_time - previous_time
            )

            previous_time = current_time

            fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            # =================================================
            # CREATE MEDIA PIPE TIMESTAMP
            # =================================================

            timestamp_ms = int(
                current_time * 1000
            )

            if timestamp_ms <= last_timestamp_ms:

                timestamp_ms = (
                    last_timestamp_ms + 1
                )

            last_timestamp_ms = timestamp_ms

            # =================================================
            # CONVERT FRAME TO RGB
            # =================================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # =================================================
            # DETECT HANDS
            # =================================================

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # =================================================
            # EXTRACT HAND INFORMATION
            # =================================================

            hands = extract_hand_information(
                result,
                frame,
            )

            # =================================================
            # CLASSIFY GESTURE
            # =================================================

            current_gesture = classifier.classify(
                hands
            )

            # =================================================
            # CONVERT GESTURE TO SAFE COMMAND
            # =================================================

            # If no hands are detected, release movement keys.
            if current_gesture == "NO_HANDS":

                controller_gesture = "NEUTRAL"

            # Unknown gestures should not activate a key.
            elif current_gesture == "UNKNOWN":

                controller_gesture = "NEUTRAL"

            else:

                controller_gesture = current_gesture

            # =================================================
            # SEND GAME COMMAND
            # =================================================

            if CONTROL_ENABLED:

                controller.execute_command(
                    controller_gesture
                )

            else:

                # Release keys when control is disabled.
                if previous_control_enabled:

                    controller.stop()

            # Store the current control state
            previous_control_enabled = (
                CONTROL_ENABLED
            )

            # =================================================
            # DISPLAY CONTROL STATUS
            # =================================================

            if CONTROL_ENABLED:

                control_status = "ENABLED"

                status_color = (
                    0,
                    255,
                    0,
                )

            else:

                control_status = "DISABLED"

                status_color = (
                    0,
                    165,
                    255,
                )

            # =================================================
            # DISPLAY GESTURE
            # =================================================

            draw_text(
                frame,
                f"Gesture: {current_gesture}",
                (20, 50),
                (0, 255, 0),
                0.9,
            )

            # =================================================
            # DISPLAY CONTROL STATUS
            # =================================================

            draw_text(
                frame,
                f"Game Control: {control_status}",
                (20, 90),
                status_color,
                0.8,
            )

            # =================================================
            # DISPLAY FPS
            # =================================================

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                (20, 130),
                (255, 255, 255),
                0.7,
            )

            # =================================================
            # DISPLAY PROFILE
            # =================================================

            draw_text(
                frame,
                f"Profile: {GAME_PROFILE}",
                (20, 155),
                (255, 255, 255),
                0.6,
            )

            # =================================================
            # DISPLAY INSTRUCTIONS
            # =================================================

            draw_text(
                frame,
                "G: Enable / Disable Control",
                (20, 195),
                (255, 255, 0),
                0.6,
            )

            draw_text(
                frame,
                "Q / ESC: Exit",
                (20, 225),
                (255, 255, 0),
                0.6,
            )

            # =================================================
            # SHOW CAMERA WINDOW
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            # =================================================
            # ENABLE / DISABLE CONTROL
            # =================================================

            if key == ord("g"):

                CONTROL_ENABLED = (
                    not CONTROL_ENABLED
                )

                if CONTROL_ENABLED:

                    print("[CONTROL] ENABLED")

                else:

                    print("[CONTROL] DISABLED")

                    # Immediately release all keys
                    controller.stop()

                    previous_control_enabled = (
                        False
                    )

            # =================================================
            # EXIT APPLICATION
            # =================================================

            elif key == ord("q") or key == 27:

                break

    except KeyboardInterrupt:

        print(
            "\nApplication interrupted."
        )

    except Exception as error:

        print(
            f"\n[ERROR] Application error: {error}"
        )

    finally:

        # ====================================================
        # SAFETY CLEANUP
        # ====================================================

        # Disable control
        CONTROL_ENABLED = False

        # Release all pressed keyboard keys
        controller.stop()

        # Close controller backend
        controller.close()

        # Release camera
        camera.release()

        # Close OpenCV windows
        cv2.destroyAllWindows()

        # Close MediaPipe landmarker
        landmarker.close()

        print(
            "Application stopped safely."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()