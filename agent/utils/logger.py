import html
import os
import sys

from . import pienv


LEVEL_SHORT_NAMES = {
    "INFO": "info",
    "ERROR": "err",
    "WARNING": "warn",
    "DEBUG": "debug",
    "CRITICAL": "critical",
    "SUCCESS": "success",
    "TRACE": "trace",
}

ANSI_LEVEL_COLORS = {
    "TRACE": "\033[34m",
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "SUCCESS": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m\033[37m",
}

HTML_LEVEL_COLORS = {
    "TRACE": "royalblue",
    "DEBUG": "deepskyblue",
    "INFO": "forestgreen",
    "SUCCESS": "forestgreen",
    "WARNING": "darkorange",
    "ERROR": "crimson",
    "CRITICAL": "firebrick",
}


def _client_name_key() -> str:
    return pienv.client_name().strip().upper()


def _is_mfaa_client() -> bool:
    return _client_name_key() == "MFAAVALONIA"


def _is_mxu_client() -> bool:
    return _client_name_key() == "MXU"


def _resolve_console_stream():
    if _is_mxu_client():
        return sys.stdout
    return sys.stderr


def _resolve_console_format() -> str:
    if _is_mfaa_client():
        return "{extra[level_short]}:{message}"
    if _is_mxu_client():
        return "{extra[mxu_html_message]}"
    return "{extra[level_color]}{message}{extra[color_reset]}"


def _short_level_name(level_name: str) -> str:
    return LEVEL_SHORT_NAMES.get(level_name, level_name.lower())


def _ansi_level_color(level_name: str) -> str:
    return ANSI_LEVEL_COLORS.get(level_name, "")


def _format_mxu_html_message(level_name: str, message: str) -> str:
    color = HTML_LEVEL_COLORS.get(level_name, "inherit")
    return f'<span style="color:{color};">{html.escape(message)}</span>'


def _enrich_record(record) -> bool:
    level_name = record["level"].name
    level_color = _ansi_level_color(level_name)

    record["extra"]["level_short"] = _short_level_name(level_name)
    record["extra"]["level_color"] = level_color
    record["extra"]["color_reset"] = "\033[0m" if level_color else ""
    record["extra"]["mxu_html_message"] = _format_mxu_html_message(
        level_name, str(record["message"])
    )
    return True


try:
    from loguru import logger as _logger

    def setup_logger(log_dir="debug/custom", console_level="INFO"):
        """设置 loguru logger

        Args:
            log_dir: 日志文件目录
            console_level: 控制台输出等级 (DEBUG, INFO, WARNING, ERROR)
        """
        os.makedirs(log_dir, exist_ok=True)
        _logger.remove()

        _logger.add(
            _resolve_console_stream(),
            format=_resolve_console_format(),
            colorize=False,
            level=console_level,
            filter=_enrich_record,
        )
        _logger.add(
            f"{log_dir}/{{time:YYYY-MM-DD}}.log",
            rotation="00:00",
            retention="2 weeks",
            compression="zip",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            filter=_enrich_record,
        )
        return _logger

    def change_console_level(level="DEBUG"):
        """动态修改控制台日志等级"""
        setup_logger(console_level=level)
        _logger.info(f"控制台日志等级已更改为: {level}")

    logger = setup_logger()
except ImportError:
    import logging

    class _ConsoleFormatter(logging.Formatter):
        def format(self, record):
            level_name = record.levelname
            message = record.getMessage()

            if _is_mfaa_client():
                return f"{_short_level_name(level_name)}:{message}"
            if _is_mxu_client():
                return _format_mxu_html_message(level_name, message)

            level_color = _ansi_level_color(level_name)
            color_reset = "\033[0m" if level_color else ""
            return f"{level_color}{message}{color_reset}"

    handler = logging.StreamHandler(_resolve_console_stream())
    handler.setFormatter(_ConsoleFormatter())

    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    logger = logging

__all__ = ["setup_logger", "change_console_level", "logger"]
