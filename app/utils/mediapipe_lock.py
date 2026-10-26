"""
backend/app/utils/mediapipe_lock.py

Central re-entrant mutex for thread-safe Google MediaPipe C++ graph execution.
Prevents CalculatorGraph packet timestamp collisions and Access Violation (0xC0000005) crashes
under concurrent requests.
"""
import threading

mediapipe_lock = threading.RLock()
