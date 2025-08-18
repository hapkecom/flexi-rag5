#
# Config settings as constants or loaded settings
#


# Constant values
maxCachedQuestions = 128
responseCacheTtlSeconds = 1*60  # 1 minute
vectorsearchEnabled = True
fulltextsearchEnabled = False
answerGradingEnabled = True
questionRewritingEnabled = True


#
# Other settings - currently NOT USED/NOT NEEDED
#
#import os

#os.environ["OPENAI_API_KEY"] = ""
#os.environ["COHERE_API_KEY"] = ""
#os.environ["TAVILY_API_KEY"] = ""

### Tracing (optional, https://docs.smith.langchain.com/)
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_API_KEY"] = ""
