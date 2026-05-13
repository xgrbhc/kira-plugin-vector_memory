"""
消息过滤模块
过滤无意义的消息，避免"流水账"问题
"""

import re
from typing import Tuple

from core.logging_manager import get_logger

logger = get_logger("vector_memory.filter", "cyan")


class MessageFilter:
    """消息过滤器"""

    # 无意义消息的正则模式
    NOISE_PATTERNS = [
        r"^[嗯啊哦呃额欸唔嘛呀]+$",  # 纯语气词
        r"^[哈呵嘿嘻]+$",  # 纯笑声
        r"^[.。，,!！?？~～]+$",  # 纯标点
        r"^(ok|OK|好的?|行|嗯+|是的?|对的?)$",  # 简单回应
        r"^(谢谢|感谢|谢啦|thx|thanks?)$",  # 简单感谢
        r"^\[.+\]$",  # 纯表情包 [xxx]
        r"^[\U0001F300-\U0001F9FF]+$",  # 纯 emoji
        r"^(666+|233+|hhhh*|hhh)$",  # 网络用语
        r"^(\?+|？+|\.+|。+)$",  # 纯符号
        r"^(晚安|早安|早上好|晚上好|你好|hello|hi|hey)$",  # 简单问候
        r"^(拜拜|再见|bye|88)$",  # 简单告别
        r"^(哦|噢|嗷|emm+|em+)$",  # 语气词
    ]

    # 中文停用词（用于计算信息密度）
    STOP_WORDS = {
        "的",
        "了",
        "是",
        "在",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "他",
        "她",
        "它",
        "们",
        "什么",
        "这个",
        "那个",
        "吗",
        "呢",
        "啊",
        "哦",
        "嗯",
        "吧",
        "呀",
        "哇",
        "诶",
        "吧",
        "啦",
        "嘞",
    }

    def __init__(self, min_meaningful_chars: int = 5, min_info_density: float = 0.3):
        """
        初始化过滤器

        Args:
            min_meaningful_chars: 最小有意义字符数
            min_info_density: 最小信息密度（非停用词占比）
        """
        self.min_meaningful_chars = min_meaningful_chars
        self.min_info_density = min_info_density
        self.noise_regex = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]

    def should_record(self, text: str) -> Tuple[bool, str]:
        """
        判断消息是否应该被记录

        Args:
            text: 消息文本

        Returns:
            (是否记录, 原因说明)
        """
        text = text.strip()

        # 1. 空文本检查
        if not text:
            return False, "空文本"

        # 2. 长度检查
        if len(text) < self.min_meaningful_chars:
            return False, f"文本过短 ({len(text)} < {self.min_meaningful_chars})"

        # 3. 噪音模式匹配
        for regex in self.noise_regex:
            if regex.match(text):
                return False, "匹配噪音模式"

        # 4. 信息密度检查（仅对中文）
        if self._contains_chinese(text):
            density = self._calculate_info_density(text)
            if density < self.min_info_density:
                return False, f"信息密度过低 ({density:.2f} < {self.min_info_density})"

        return True, "通过过滤"

    def _contains_chinese(self, text: str) -> bool:
        """检查是否包含中文"""
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _calculate_info_density(self, text: str) -> float:
        """
        计算信息密度（非停用词占比）
        """
        # 简单分词（按单字符）
        chars = [c for c in text if c.strip() and "\u4e00" <= c <= "\u9fff"]

        if not chars:
            return 1.0  # 非中文文本默认高密度

        # 计算非停用词占比
        non_stop_chars = [c for c in chars if c not in self.STOP_WORDS]

        return len(non_stop_chars) / len(chars) if chars else 0

    def extract_key_info(self, text: str) -> str:
        """
        提取关键信息（用于优化存储）

        Args:
            text: 原始文本

        Returns:
            提取后的关键信息
        """
        # 移除多余空白
        text = re.sub(r"\s+", " ", text).strip()

        # 移除重复标点
        text = re.sub(r"([。！？,.!?])\1+", r"\1", text)

        # 移除开头的 @ 和回复标记
        text = re.sub(r"^@\S+\s*", "", text)

        return text

    def get_text_stats(self, text: str) -> dict:
        """
        获取文本统计信息

        Args:
            text: 文本内容

        Returns:
            统计信息字典
        """
        text = text.strip()

        chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        non_chinese = len(text) - len(chinese_chars)

        return {
            "total_length": len(text),
            "chinese_chars": len(chinese_chars),
            "non_chinese_chars": non_chinese,
            "info_density": (
                self._calculate_info_density(text) if chinese_chars else 1.0
            ),
            "has_numbers": bool(re.search(r"\d", text)),
            "has_urls": bool(re.search(r"https?://", text)),
        }
