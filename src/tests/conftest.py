"""テストから src 直下のモジュールを import できるようにする。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
