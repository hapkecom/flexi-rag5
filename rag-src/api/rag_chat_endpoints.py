# langserve doesn’t have a built-in function for OpenAI compatibility
# create custom endpoints manually:

import json
import logging
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
import shortuuid
#from rag_workflow.simple_chat_workflow import create_workflow
from rag_workflow.chat_workflow import create_workflow
from langchain_core.chat_history import BaseChatMessageHistory



logger = logging.getLogger(__name__)

router = APIRouter()
workflow = create_workflow()

class ChatRequest(BaseModel):
    model: str
    stream: Optional[bool] = False
    messages: list

class EmbeddingsRequest(BaseModel):
    input: list
    model: str

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Example request:
    {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": The Los Angeles Dodgers won the World Series in 2020. Where was it played?"
            }
        ]
    }

    Example response (Non-Streaming):
    {
        "choices": [
            {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                "role": "assistant"
            },
            "logprobs": null
            }
        ],
        "created": 1677664795,
        "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
        "model": "gpt-4o-mini",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 17,
            "prompt_tokens": 57,
            "total_tokens": 74
        }
    }
    """

    try:
        # Some "settings" in the response
        id = "chatcmpl-flexirag-"+str(shortuuid.uuid()[:7])
        model = "flexirag-v1"

        # Generate response
        stream = request.stream
        if not stream:
            # Non-streaming mode
            async def generate():
                workflow_result = await workflow.ainvoke({"messages": request.messages})  # Adjust according to your workflow execution method
                response_data = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "message": {
                            "content": workflow_result['generation'].content,
                            "role": "assistant"
                        }
                    }],
                    "created": int(time.time()),
                }
                return response_data
            return await generate()

        else:
            # Steaming mode,
            # Implemantation based on ideas of: https://www.digitalocean.com/community/tutorials/python-str-repr-functions
            inputs = {"messages": request.messages, "stream_generate_on_last_node": stream}
            async def generate():
                async for workflow_event in workflow.astream_events(inputs, version="v2"):
                    kind = workflow_event["event"]
                    tags = workflow_event.get("tags", [])
                    if kind == "on_chat_model_stream" and "final_node" in tags:
                        # fetch "content"
                        data = workflow_event["data"]
                        if data is None:
                            continue
                        chunk = data["chunk"]
                        if chunk is None:
                            continue
                        content = chunk.content
                        if content is None:
                            continue
                        # process content: i.e. send to REST client
                        logger.info(f"Final node content (Chunk): '{content}'")
                        # {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}
                        response_data = {
                            "id": id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content,
                                        "role": "assistant",
                                    },
                                    "finish_reason": None,
                                },
                            ],
                            "created": int(time.time()),
                        }
                        # yield as Server-Side-Event (SSE)
                        data_str = f"data: {json.dumps(response_data)}"
                        data_str_with_newlines = data_str +"\n\n"
                        logger.debug(f"  Yielding (plus non-logged newlines): {data_str}")
                        yield data_str_with_newlines

                # Generation finished
                stop_response_data = {
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ]
                }
                yield f"data: {json.dumps(stop_response_data)}\n\n"
                # (response) stream finished
                yield f"data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest):
    try:
        embeddings = generate_embeddings(request.input, request.model)
        return {"data": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_embeddings(inputs, model):
    # Logic for generating embeddings using the specified model
    embeddings = []
    for input in inputs:
        embedding = call_embedding_api(input, model)
        embeddings.append(embedding)
    return embeddings

def call_embedding_api(input, model):
    # Example function to call an external embedding API
    return {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
