import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from typing import Any, Dict, List, Literal, Optional, Union
from openai import OpenAI

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BaseLLMClient:
    NAME = "BaseLLMClient"

    def __init__(
        self,
        max_attempt: int = 2,
    ) -> None:
        self._llm_client = OpenAI(
            api_key=os.getenv("API_KEY", "OLLAMA"),
            base_url=os.getenv("BASE_URL"),
        )
        self._llm_config = {
            "model": os.getenv("MODEL"),
            "temperature": 0.7,
            "max_tokens": 8192,
        }
        self._max_attempt = max_attempt
        assert (
            max_attempt >= 1
        ), f"max_attempt should be no less than 1 (but {max_attempt} was given)!"

    def generate_content_with_messages(self, messages: List[dict], **llm_config) -> str:
        """
        根据消息和配置生成内容。
        如果 llm_config 中未提供某些参数，则使用默认值。
        """
        config = {**self._llm_config, **llm_config}

        content = self._get_cache(messages, config)
        if content is False or content is None or content == "":
            response = self._get_response_with_messages(messages, **config)

            if response is None:
                raise ValueError(
                    f"Response is None. Please check your LLM client configuration."
                )
            else:
                content = self._get_content_from_response(response)

            self._save_cache(messages, config, content)
        return content

    def _get_cache(self, messages: List[dict], config: dict) -> Union[str, bool]:
        """
        获取缓存内容
        """
        return False

    def _save_cache(self, messages: List[dict], config: dict, content: str) -> None:
        """
        保存缓存内容
        """
        pass

    def _get_response_with_messages(
        self,
        messages: List[dict],
        **llm_config: Dict[str, Any],
    ) -> Optional[dict]:
        """
        获取响应
        """
        model = llm_config.get("model")
        temperature = llm_config.get("temperature")
        max_tokens = llm_config.get("max_tokens")

        for attempt in range(self._max_attempt):
            try:
                response = self._llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self._max_attempt - 1:
                    raise e

    def _get_content_from_response(self, response: dict) -> str:
        """
        从响应中获取内容
        """
        return response.choices[0].message.content
