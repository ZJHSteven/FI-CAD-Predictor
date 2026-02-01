import sys
import os
import atexit
from .config_loader import create_config_loader

class StdoutRedirector:
    """
    自动从配置文件获取日志路径，重定向stdout和stderr到日志文件。
    """
    def __init__(self):
        # 自动获取日志路径
        config_loader = create_config_loader()
        paths_config = config_loader.get_paths_config()
        log_path = paths_config['logs']['main_log']
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")
        # 重定向
        sys.stdout = self.log_file
        sys.stderr = self.log_file
        # 程序退出时自动关闭
        atexit.register(self.close)

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass
