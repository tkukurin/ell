from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union
from ell.provider import APICallResult, Provider
from ell.types import Message, ContentBlock, ToolCall
from ell.types._lstr import _lstr
import json
from ell.configurator import config, register_provider
from ell.types.message import LMP
from ell.util.serialization import serialize_image

try: 
    import google.generativeai as genai
    class GeminiProvider(Provider):

        @staticmethod
        def fmt_block(content_block: ContentBlock) -> Dict[str, Any]:
            if content_block.image:
                raise NotImplemented()
            elif content_block.text:
                return {
                    "type": "text",
                    "text": content_block.text
                }
            elif content_block.parsed:
                return {
                    "type": "text",
                    "text": content_block.parsed.model_dump_json()
                }
            return None

        @staticmethod
        def fmt(message: Message) -> Dict[str, Any]:
            role = "model" if message.role == 'system' else message.role
            openai_message = {
                "role": "tool" if message.tool_results else role,
                "content": list(filter(None, [
                    GeminiProvider.fmt_block(c) for c in message.content
                ]))
            }
            if message.tool_calls:
                raise NotImplementedError()
            if message.tool_results:
                raise NotImplementedError()
            return openai_message

        @classmethod
        def call_model(
            cls,
            client: genai.GenerativeModel,
            model: str,
            messages: List[Message],
            api_params: Dict[str, Any],
            tools: Optional[list[LMP]] = None,
        ) -> APICallResult:
            final_call_params = api_params.copy()
            openai_messages = [
                cls.fmt(message)
                for message in messages
            ]

            actual_n = api_params.get("n", 1)
            final_call_params["model"] = model
            final_call_params["messages"] = openai_messages
            chat = client.start_chat(
                history=[
                    {'role': 'model' if m.role == 'system' else m.role, 'parts': [
                        c.text for m in messages[:-1] for c in m.content
                    ]}
                    for m in messages[:-1]
                ],
            )
            response = chat.send_message(
                '\n'.join(c.text for c in messages[-1].content)
            )

            return APICallResult(
                response=response,
                actual_streaming=False,
                actual_n=actual_n,
                final_call_params=final_call_params,
            )

        @classmethod
        def process_response(
            cls, call_result: APICallResult, _invocation_origin: str,  logger : Optional[Any] = None,  tools: Optional[List[LMP]] = None,
        ) -> Tuple[List[Message], Dict[str, Any]]:
            choices_progress = defaultdict(list)
            api_params = call_result.final_call_params
            metadata = {}

            # TODO ugly adapter idk that's how openai does it
            # then in ell.simple it just wraps
            class WithContent(NamedTuple):
                content: list
            class WithText(NamedTuple):
                text: str

            # what we need is structure of
            # tracked_results[0].content[0].content[0].text
            tracked_results = WithContent(content=
                [WithContent(content=[WithText(x.text)]) 
                 for x in call_result.response.parts]
            )
            return tracked_results, metadata

        @classmethod
        def supports_streaming(cls) -> bool:
            return True

        @classmethod
        def get_client_type(cls) -> Type:
            return genai.GenerativeModel


    register_provider(GeminiProvider)
except ImportError:
    pass