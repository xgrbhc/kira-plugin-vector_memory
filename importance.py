"""
重要性评分模块
支持规则评分和 LLM 评分两种模式
"""

import re
from abc import ABC, abstractmethod
from typing import Optional

from core.logging_manager import get_logger

logger = get_logger("vector_memory.importance", "cyan")


# 重要性关键词库
IMPORTANCE_KEYWORDS = {
    "high": {
        "keywords": [
            "喜欢",
            "讨厌",
            "过敏",
            "名字叫",
            "我叫",
            "生日",
            "地址",
            "电话",
            "记住",
            "重要",
            "约定",
            "承诺",
            "永远",
            "一直",
            "习惯",
            "每天",
            "工作",
            "职业",
            "专业",
            "爱好",
            "兴趣",
            "家人",
            "朋友",
            "目标",
            "梦想",
            "计划",
            "决定",
            "害怕",
            "恐惧",
            "不能吃",
            "不喜欢",
        ],
        "score": 0.85,
    },
    "medium": {
        "keywords": [
            "最近",
            "打算",
            "想要",
            "希望",
            "觉得",
            "认为",
            "感觉",
            "项目",
            "学习",
            "研究",
            "尝试",
            "正在",
            "准备",
            "需要",
        ],
        "score": 0.5,
    },
    "low": {
        "keywords": [
            "在吗",
            "好的",
            "嗯嗯",
            "哈哈",
            "emmm",
            "啊啊",
            "ok",
            "666",
            "哦哦",
            "是的",
            "对的",
            "知道了",
            "收到",
            "了解",
            "明白",
        ],
        "score": 0.1,
    },
}


class BaseImportanceScorer(ABC):
    """重要性评分基类"""

    @abstractmethod
    async def score(self, text: str) -> float:
        """
        为文本内容评分

        Args:
            text: 文本内容

        Returns:
            重要性分数 (0.0 - 1.0)
        """
        pass


class RuleBasedScorer(BaseImportanceScorer):
    """基于规则的重要性评分"""

    def __init__(self):
        self.default_score = 0.3

    async def score(self, text: str) -> float:
        """基于关键词匹配评分"""
        text_lower = text.lower()

        # 检查高优先级关键词
        for keyword in IMPORTANCE_KEYWORDS["high"]["keywords"]:
            if keyword in text_lower:
                logger.debug(f"匹配高优先级关键词: {keyword}")
                return IMPORTANCE_KEYWORDS["high"]["score"]

        # 检查中优先级关键词
        for keyword in IMPORTANCE_KEYWORDS["medium"]["keywords"]:
            if keyword in text_lower:
                return IMPORTANCE_KEYWORDS["medium"]["score"]

        # 检查低优先级/噪音关键词
        for keyword in IMPORTANCE_KEYWORDS["low"]["keywords"]:
            if keyword in text_lower or text_lower == keyword:
                return IMPORTANCE_KEYWORDS["low"]["score"]

        # 额外规则：长文本可能更重要
        if len(text) > 100:
            return min(self.default_score + 0.2, 0.6)

        # 额外规则：包含数字可能是具体信息
        if re.search(r"\d{3,}", text):  # 3位以上数字
            return min(self.default_score + 0.15, 0.5)

        return self.default_score


class LLMBasedScorer(BaseImportanceScorer):
    """基于 LLM 的重要性评分"""

    SCORING_PROMPT = """请评估以下对话内容的记忆重要性（0.0-1.0）：

评分标准：
- 0.9-1.0: 用户核心身份信息、重要偏好、健康/安全相关
- 0.7-0.8: 用户计划、目标、重要关系信息
- 0.4-0.6: 一般性话题讨论、日常对话
- 0.1-0.3: 无实质内容的闲聊、简单回应

对话内容：
{text}

仅返回一个 0.0-1.0 之间的数字，如: 0.7"""

    def __init__(self, llm_api, model: Optional[str] = None):
        self.llm_api = llm_api
        self.model = model

    async def score(self, text: str) -> float:
        """调用 LLM 评分"""
        try:
            prompt = self.SCORING_PROMPT.format(text=text[:500])  # 限制长度

            from core.provider import LLMRequest
            request = LLMRequest(messages=[{"role": "user", "content": prompt}])
            
            if self.model and ":" in self.model:
                provider_id, _, model_id = self.model.partition(":")
                llm_client = self.llm_api.provider_mgr.get_model_client(provider_id, model_id)
            else:
                llm_client = self.llm_api.provider_mgr.get_default_llm()
                
            response = await llm_client.chat(request)

            # 解析响应
            score_text = response.text_response.strip()
            # 尝试提取数字
            match = re.search(r"(\d+\.?\d*)", score_text)
            if match:
                score = float(match.group(1))
            else:
                score = float(score_text)

            # 确保在有效范围内
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"LLM 评分失败，使用默认值: {e}")
            return 0.3


class HybridScorer(BaseImportanceScorer):
    """混合评分：规则 + LLM"""

    def __init__(self, llm_api, model: Optional[str] = None):
        self.rule_scorer = RuleBasedScorer()
        self.llm_scorer = LLMBasedScorer(llm_api, model)

        # 仅对规则评分为中等的内容使用 LLM 细化
        self.llm_threshold_low = 0.25
        self.llm_threshold_high = 0.75

    async def score(self, text: str) -> float:
        """混合评分策略"""
        rule_score = await self.rule_scorer.score(text)

        # 明确高或低的直接返回规则分数
        if (
            rule_score <= self.llm_threshold_low
            or rule_score >= self.llm_threshold_high
        ):
            return rule_score

        # 中间区域使用 LLM 细化
        try:
            llm_score = await self.llm_scorer.score(text)
            # 加权平均
            return rule_score * 0.4 + llm_score * 0.6
        except Exception:
            return rule_score


class ImportanceScorerFactory:
    """评分器工厂"""

    @staticmethod
    def create(
        mode: str, llm_api=None, model: Optional[str] = None
    ) -> BaseImportanceScorer:
        """
        创建评分器实例

        Args:
            mode: 评分模式 (rule/llm/hybrid)
            llm_api: LLM 客户端（llm/hybrid 模式需要）
            model: 指定模型

        Returns:
            BaseImportanceScorer 实例
        """
        if mode == "rule":
            return RuleBasedScorer()
        elif mode == "llm":
            if not llm_api:
                raise ValueError("LLM 评分模式需要 llm_api")
            return LLMBasedScorer(llm_api, model)
        elif mode == "hybrid":
            if not llm_api:
                raise ValueError("混合评分模式需要 llm_api")
            return HybridScorer(llm_api, model)
        else:
            logger.warning(f"未知评分模式 {mode}，使用规则模式")
            return RuleBasedScorer()
