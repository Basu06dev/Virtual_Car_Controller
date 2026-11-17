
"""
Windows SendInput keyboard backend.

Profiles:
    default -> Arrow keys
    wasd    -> W, A, S, D
    racing  -> Arrow keys

Commands:
    LEFT
    RIGHT
    FORWARD
    STOP
    NEUTRAL
"""

import time
import ctypes

from abc import ABC, abstractmethod
from ctypes import wintypes


# ============================================================
# WINDOWS CONSTANTS
# ============================================================

INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001


# ============================================================
# WINDOWS POINTER TYPE
# ============================================================

# ULONG_PTR is a pointer-sized unsigned integer.
# ctypes.wintypes does not always expose ULONG_PTR.

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


# ============================================================
# WINDOWS INPUT STRUCTURES
# ============================================================

class MOUSEINPUT(ctypes.Structure):

    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):

    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):

    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):

    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):

    _anonymous_ = ("union",)

    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


# ============================================================
# WINDOWS VIRTUAL KEY CODES
# ============================================================

VIRTUAL_KEYS = {

    # Alphabet keys
    "a": 0x41,
    "b": 0x42,
    "c": 0x43,
    "d": 0x44,
    "e": 0x45,
    "f": 0x46,
    "g": 0x47,
    "h": 0x48,
    "i": 0x49,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "m": 0x4D,
    "n": 0x4E,
    "o": 0x4F,
    "p": 0x50,
    "q": 0x51,
    "r": 0x52,
    "s": 0x53,
    "t": 0x54,
    "u": 0x55,
    "v": 0x56,
    "w": 0x57,
    "x": 0x58,
    "y": 0x59,
    "z": 0x5A,

    # Arrow keys
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,

    # Other keys
    "space": 0x20,
    "enter": 0x0D,
    "escape": 0x1B,
}


# ============================================================
# VALID ACTIONS
# ============================================================

VALID_ACTIONS = {
    "NEUTRAL",
    "LEFT",
    "RIGHT",
    "FORWARD",
    "STOP",
}


# ============================================================
# GAME PROFILES
# ============================================================

GAME_PROFILES = {

    "default": {
        "LEFT": "left",
        "RIGHT": "right",
        "FORWARD": "up",
        "STOP": "down",
    },

    "wasd": {
        "LEFT": "a",
        "RIGHT": "d",
        "FORWARD": "w",
        "STOP": "s",
    },

    "racing": {
        "LEFT": "left",
        "RIGHT": "right",
        "FORWARD": "up",
        "STOP": "down",
    },

}


# ============================================================
# INPUT BACKEND
# ============================================================

class InputBackend(ABC):

    @abstractmethod
    def press(self, key):
        """Press and hold a key."""
        pass

    @abstractmethod
    def release_all(self):
        """Release all currently pressed keys."""
        pass

    @abstractmethod
    def close(self):
        """Close the input backend."""
        pass


# ============================================================
# WINDOWS SENDINPUT BACKEND
# ============================================================

class WindowsSendInput(InputBackend):
    """
    Keyboard input using the Windows SendInput API.

    Input is sent to the active foreground window.
    """

    def __init__(self):

        self.pressed_keys = set()

        self.user32 = ctypes.WinDLL(
            "user32",
            use_last_error=True
        )

        self.user32.SendInput.argtypes = [
            wintypes.UINT,
            ctypes.POINTER(INPUT),
            ctypes.c_int,
        ]

        self.user32.SendInput.restype = wintypes.UINT

    # --------------------------------------------------------
    # GET VIRTUAL KEY CODE
    # --------------------------------------------------------

    def _get_virtual_key(self, key):

        key = str(key).lower().strip()

        if key not in VIRTUAL_KEYS:

            raise ValueError(
                f"Unsupported key: {key}"
            )

        return VIRTUAL_KEYS[key]

    # --------------------------------------------------------
    # SEND KEY EVENT
    # --------------------------------------------------------

    def _send_key(self, key, key_up=False):

        key = str(key).lower().strip()

        virtual_key = self._get_virtual_key(key)

        flags = 0

        # Release key
        if key_up:

            flags |= KEYEVENTF_KEYUP

        # Arrow keys are extended keys
        if key in {
            "left",
            "right",
            "up",
            "down",
        }:

            flags |= KEYEVENTF_EXTENDEDKEY

        keyboard_input = KEYBDINPUT(
            wVk=virtual_key,
            wScan=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )

        input_union = INPUT_UNION(
            ki=keyboard_input
        )

        input_event = INPUT(
            type=INPUT_KEYBOARD,
            union=input_union,
        )

        result = self.user32.SendInput(
            1,
            ctypes.byref(input_event),
            ctypes.sizeof(INPUT),
        )

        if result != 1:

            error_code = ctypes.get_last_error()

            raise ctypes.WinError(
                error_code
            )

    # --------------------------------------------------------
    # PRESS KEY
    # --------------------------------------------------------

    def press(self, key):

        key = str(key).lower().strip()

        if key in self.pressed_keys:

            return

        self._send_key(
            key,
            key_up=False
        )

        self.pressed_keys.add(key)

    # --------------------------------------------------------
    # RELEASE ALL KEYS
    # --------------------------------------------------------

    def release_all(self):

        for key in list(self.pressed_keys):

            try:

                self._send_key(
                    key,
                    key_up=True
                )

            except Exception as error:

                print(
                    f"[INPUT WARNING] "
                    f"Could not release {key}: {error}"
                )

        self.pressed_keys.clear()

    # --------------------------------------------------------
    # CLOSE BACKEND
    # --------------------------------------------------------

    def close(self):

        self.release_all()


# ============================================================
# GAME CONTROLLER
# ============================================================

class GameController:
    """
    Universal gesture game controller.

    Parameters:
        backend:
            Input backend. Defaults to WindowsSendInput.

        profile:
            default, wasd, or racing.

        key_mapping:
            Optional custom command-to-key mapping.
    """

    def __init__(
        self,
        backend=None,
        profile="default",
        key_mapping=None,
    ):

        if backend is not None:

            self.backend = backend

        else:

            self.backend = WindowsSendInput()

        self.current_command = "NEUTRAL"
        self.current_key = None

        self.profile = None
        self.key_mapping = {}

        if key_mapping is not None:

            self.set_custom_mapping(
                key_mapping
            )

        else:

            self.set_profile(
                profile
            )

    # ========================================================
    # PROFILE MANAGEMENT
    # ========================================================

    def set_profile(self, profile):

        profile = str(
            profile
        ).lower().strip()

        if profile not in GAME_PROFILES:

            available_profiles = ", ".join(
                GAME_PROFILES.keys()
            )

            raise ValueError(
                f"Unknown profile: {profile}. "
                f"Available profiles: {available_profiles}"
            )

        # Release any existing keys
        self.stop()

        self.profile = profile

        self.key_mapping = GAME_PROFILES[
            profile
        ].copy()

        print(
            f"[CONTROLLER] Profile selected: "
            f"{self.profile}"
        )

        print(
            f"[CONTROLLER] Key mapping: "
            f"{self.key_mapping}"
        )

    # ========================================================
    # CUSTOM MAPPING
    # ========================================================

    def set_custom_mapping(self, key_mapping):

        required_commands = {
            "LEFT",
            "RIGHT",
            "FORWARD",
            "STOP",
        }

        missing_commands = (
            required_commands
            - set(key_mapping.keys())
        )

        if missing_commands:

            raise ValueError(
                "Missing commands: "
                f"{missing_commands}"
            )

        self.stop()

        self.profile = "custom"

        self.key_mapping = key_mapping.copy()

        print(
            "[CONTROLLER] Custom key mapping selected"
        )

        print(
            f"[CONTROLLER] Key mapping: "
            f"{self.key_mapping}"
        )

    # ========================================================
    # AVAILABLE PROFILES
    # ========================================================

    def get_available_profiles(self):

        return list(
            GAME_PROFILES.keys()
        )

    # ========================================================
    # COMMAND EXECUTION
    # ========================================================

    def execute_command(self, command):

        if command is None:

            command = "NEUTRAL"

        command = str(
            command
        ).upper().strip()

        if command not in VALID_ACTIONS:

            print(
                f"[CONTROLLER WARNING] "
                f"Invalid command: {command}"
            )

            return

        # Avoid repeating the same command
        if command == self.current_command:

            return

        # Release previous keys
        self.backend.release_all()

        self.current_key = None

        # Neutral releases all movement keys
        if command == "NEUTRAL":

            self.current_command = "NEUTRAL"

            print(
                "[CONTROLLER] Action: NEUTRAL"
            )

            return

        # Find key for command
        key = self.key_mapping.get(
            command
        )

        if key is None:

            print(
                f"[CONTROLLER WARNING] "
                f"No key mapping for {command}"
            )

            self.current_command = "NEUTRAL"

            return

        try:

            self.backend.press(
                key
            )

            self.current_key = key

            self.current_command = command

            print(
                f"[CONTROLLER] Action: "
                f"{command} -> Key: {key}"
            )

        except Exception as error:

            print(
                f"[CONTROLLER ERROR] "
                f"Could not press {key}: {error}"
            )

            self.current_command = "NEUTRAL"

    # ========================================================
    # SAFETY STOP
    # ========================================================

    def stop(self):

        self.backend.release_all()

        self.current_key = None

        self.current_command = "NEUTRAL"

        print(
            "[CONTROLLER] All inputs released"
        )

    # ========================================================
    # CLOSE CONTROLLER
    # ========================================================

    def close(self):

        self.stop()

        self.backend.close()

        print(
            "[CONTROLLER] Controller closed"
        )


# ============================================================
# STANDALONE TEST
# ============================================================

def standalone_test():

    print()
    print("========================================")
    print(" WINDOWS SENDINPUT CONTROLLER TEST")
    print("========================================")
    print()

    controller = None

    try:

        print("Available profiles:")

        for profile in GAME_PROFILES:

            print(
                f"  - {profile}"
            )

        print()

        # Choose the profile to test.
        #
        # default -> Arrow keys
        # wasd    -> W, A, S, D
        # racing  -> Arrow keys

        selected_profile = "default"

        controller = GameController(
            profile=selected_profile
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "Click the target game window "
            "before the test begins."
        )

        print(
            "The test will begin in 3 seconds..."
        )

        time.sleep(3)

        # Test LEFT
        controller.execute_command(
            "LEFT"
        )

        time.sleep(1)

        # Test RIGHT
        controller.execute_command(
            "RIGHT"
        )

        time.sleep(1)

        # Test FORWARD
        controller.execute_command(
            "FORWARD"
        )

        time.sleep(1)

        # Test STOP
        controller.execute_command(
            "STOP"
        )

        time.sleep(1)

        # Test NEUTRAL
        controller.execute_command(
            "NEUTRAL"
        )

        time.sleep(1)

        # Release all keys
        controller.stop()

        print()

        print(
            "Controller test completed."
        )

    except KeyboardInterrupt:

        print(
            "\n[TEST] Test interrupted."
        )

    except Exception as error:

        print(
            f"\n[ERROR] {error}"
        )

    finally:

        if controller is not None:

            controller.close()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    standalone_test()