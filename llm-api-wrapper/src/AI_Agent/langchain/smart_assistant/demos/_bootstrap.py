"""
demo 公共引导（补充时间: 2026-07-23）

把 smart_assistant 的父目录加入 sys.path，使各 demo 能以
`from smart_assistant.xxx import ...` 的方式导入，无论是脚本还是模块方式运行。
"""

import os
import sys


def setup_path():
    here = os.path.dirname(os.path.abspath(__file__))        # .../smart_assistant/demos
    smart_assistant_dir = os.path.dirname(here)              # .../smart_assistant
    parent = os.path.dirname(smart_assistant_dir)            # .../langchain
    if parent not in sys.path:
        sys.path.insert(0, parent)
