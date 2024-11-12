import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

class CustomLogger:
    _instance: Optional[logging.Logger] = None
    _initialized: bool = False

    @classmethod
    def setup_logger(cls, debug_mode: bool = False) -> None:
        """初始化日志系统"""
        if cls._initialized:
            # 如果已经初始化，只更新日志级别
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
            cls.update_log_level(debug_mode)
            return

        # 创建日志目录
        os.makedirs('logs', exist_ok=True)

        # 配置第三方库的日志级别
        third_party_loggers = [
            'openai',
            'urllib3',
            'requests',
            'httpcore'
        ]
        
        for logger_name in third_party_loggers:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)  # 将第三方库的日志级别设置为 WARNING
        
        # 获取根日志记录器
        root_logger = logging.getLogger()
        
        # 设置日志级别
        root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        # 如果已经有处理器，先清除
        if root_logger.handlers:
            root_logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
        )
        console_handler.setFormatter(console_format)

        # 文件处理器
        file_handler = RotatingFileHandler(
            'logs/app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # 文件始终记录所有级别的日志
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_format)

        # 添加处理器
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if not cls._initialized:
            cls.setup_logger()
        return logging.getLogger(name)

    @classmethod
    def update_log_level(cls, debug_mode: bool) -> None:
        """更新日志级别"""
        if not cls._initialized:
            cls.setup_logger(debug_mode)
            return
            
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
        # 确保第三方库的日志级别保持在 WARNING
        third_party_loggers = [
            'openai',
            'urllib3',
            'requests',
            'httpcore'
        ]
        
        for logger_name in third_party_loggers:
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.setLevel(logging.WARNING)
            
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

logger = CustomLogger.get_logger("app")