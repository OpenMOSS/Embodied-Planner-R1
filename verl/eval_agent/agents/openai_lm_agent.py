import openai
import logging
# import backoff
import random
from .base import LMAgent

logger = logging.getLogger("agent_frame")


class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        if "api_base" in config:
            # openai.api_base = config['api_base']
            self.api_base = config['api_base']
        if "api_key" in config:
            # openai.api_key = config['api_key']
            self.api_key = config['api_key']

    def __call__(self, messages) -> str:
        # Prepend the prompt with the system message
        retry_limit = 20
        while True:
            try:
                if isinstance(self.api_key, list):
                    api_key = random.choice(self.api_key)
                else:
                    api_key = self.api_key
                client = openai.OpenAI(
                    base_url=self.api_base,
                    api_key=api_key
                )
                if 'gpt' in self.config['model_name'] :
                    response = client.chat.completions.create(
                        model=self.config["model_name"],
                        messages=messages,
                        max_tokens=self.config.get("max_tokens", 512),
                        temperature=self.config.get("temperature", 0),
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.config["model_name"],
                        messages=messages,
                        max_tokens=self.config.get("max_tokens", 512),
                        temperature=self.config.get("temperature", 0),
                        max_completion_tokens=self.config.get("max_completion_tokens", 512),
                    )
                # import pdb;pdb.set_trace()
                return response.choices[0].message.content
            except Exception as e:
                if isinstance(self.api_key, list) and "your account balance is insufficient" in e:
                        self.api_key.remove(api_key)
                        continue

                if retry_limit >= 0:
                    retry_limit -= 1
                    print(f"Error: {e}, retry times {str(20-retry_limit)}", flush=True)
                    continue
                
                raise e
                

