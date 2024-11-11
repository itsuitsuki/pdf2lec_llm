import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

class CustomLogger:
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "app") -> logging.Logger:
        if cls._instance is None:
            cls._instance = cls._setup_logger(name)
        return cls._instance

    @staticmethod
    def _setup_logger(name: str) -> logging.Logger:
        os.makedirs('logs', exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)

            file_handler = RotatingFileHandler(
                'logs/app.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.INFO)
            file_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(file_format)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

logger = CustomLogger.get_logger()