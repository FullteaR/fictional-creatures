"""テストから web/ を import できるようにし、pipeline の入出力を一時ディレクトリに向ける。"""
import os
import sys
import tempfile

WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEB_DIR)

os.environ.setdefault("PIPELINE_DIR", os.path.join(ROOT, "src"))
os.environ.setdefault("ENDEMIC_OUT_DIR", tempfile.mkdtemp(prefix="endemic-tests-"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, WEB_DIR)
