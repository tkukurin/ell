from ell.configurator import config
import google.generativeai as genai

import logging
import colorama

import os
logger = logging.getLogger(__name__)

class Client:
    """adapter for gemini
    """

    def __init__(self):
        self.api_key = os.environ["GEMINI_API_KEY"]
    
    def call_model(self, *a, **kw):
        print("-----------------")
        print(f"{a=}")
        print(f"{kw=}")
        print("-----------------")
        return self(*a, **kw)

    def __call__(
        self,
        # model_name: str = "gemini-1.5-flash",
        message: str = "",
        **kwargs
    ):
        print("-----------------")
        print(message)
        print(kwargs)
        print("-----------------")

        model = genai.GenerativeModel(
          generation_config=(generation_config := {
              "temperature": 1,
              "top_p": 0.95,
              "top_k": 64,
              "max_output_tokens": 8192,
              "response_mime_type": "text/plain",
          })
        )
        session = model.start_chat(history=[])
        return session.send_message(message)


def register(client: genai.GenerativeModel):
    """Register gemini
    """
    model_data = [
        ('gemini-1.5-flash', 'system'),
    ]
    for model_id, owned_by in model_data:
        config.register_model(model_id, client)


# default_client = Client() #  genai.GenerativeModel
default_client = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=None,
)
setattr(default_client, "api_key", os.environ["GEMINI_API_KEY"])
register(default_client)
config.default_client = default_client