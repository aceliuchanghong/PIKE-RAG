from dataclasses import dataclass
from typing import Any, Dict, List
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from myrag.my_prompt.base_parser import BaseParser
from myrag.my_prompt.message_template import MessageTemplate


@dataclass
class CommunicationProtocol:
    template: MessageTemplate
    parser: BaseParser

    def template_partial(self, **kwargs) -> List[str]:
        """Partially fill in the template placeholders to update the template.

        Args:
            **kwargs: the key, value pairs for the partially fill in variables.

        Returns:
            List[str]: the remaining input variables needed to fill in for the updated template.
        """
        self.template = self.template.partial(**kwargs)
        return self.template.input_variables

    def process_input(self, content: str, **kwargs) -> List[Dict[str, str]]:
        """Fill in the placeholders in the message template to form an input message list.

        Args:
            content (str): the main content for encoding.
            kwargs (dict): the optional key-value pairs that may be used for encoding.

        Returns:
            List[Dict[str, str]]: the formatted message list for LLM chat.
        """
        encoded_content, encoded_dict = self.parser.encode(content, **kwargs)
        return self.template.format(content=encoded_content, **kwargs, **encoded_dict)

    def parse_output(self, content: str, **kwargs) -> Any:
        """Let the parser to decode the response content.

        Args:
            content (str): the main content for parsing.
            kwargs (dict): the optional key-value pairs that may be used for parsing.

        Returns:
            Any: value(s) returned by the parser, the return value types varied according to different applications.
        """
        return self.parser.decode(content, **kwargs)


if __name__ == "__main__":
    """
    uv run myrag/my_prompt/protocol.py
    """
    template = MessageTemplate(template=[("user", "请回答问题：{content}")])
    parser = BaseParser()
    protocol = CommunicationProtocol(template, parser)

    messages = protocol.process_input("北京的首都在哪里？")
    # 输出：[{"role": "user", "content": "请回答问题：北京的首都在哪里？"}]
    print(f"{messages}")

    # 假设获得了LLM的响应
    response = "北京就是中国的首都"
    # 解析输出
    result = protocol.parse_output(response)
    print(f"{result}")
    # 输出：北京就是中国的首都
