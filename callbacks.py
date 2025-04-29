from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Handle the start of an LLM call."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Handle the start of an LLM call."""
        pass