import logging                          
logger = logging.getLogger(__name__)    

class LLMWithFallback:
    def __init__(self, primary_llm, fallback_llm):
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.using_fallback = False

    def invoke(self, messages):
        try:
            response = self.primary.invoke(messages)
            self.using_fallback = False
            return response
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                logger.warning("Gemini unavailable (503), switching to Ollama fallback")
                self.using_fallback = True
            elif "API_KEY_INVALID" in str(e) or "INVALID_ARGUMENT" in str(e):
                logger.error("Gemini API key invalid — using Ollama fallback. Check your API key.")
                self.using_fallback = True
            else:
                raise

            try:
                return self.fallback.invoke(messages)
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                self.using_fallback = False
                raise Exception("Both models unavailable. Please try again later.")
           