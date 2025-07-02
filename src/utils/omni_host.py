import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
import asyncio
import base64
import io
import uvicorn
from typing import Any, Dict, List

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


model_path = './model/qwen2.5-omni-7b'
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

request_queue: asyncio.Queue = asyncio.Queue()

async def worker():
    while True:
        task = await request_queue.get()
        try:
            data, future = task["data"], task["future"]
            conversation = data.get("conversation")
            use_audio = data.get("use_audio_in_video", True)

            # 准备输入
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=use_audio
            )
            print('1111')
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            input_ids = inputs.input_ids
            print('2222')
            # 推理
            text_ids = model.generate(
                **inputs, use_audio_in_video=use_audio, return_audio=False
            )
            # text_ids = text_ids[:, input_ids.shape[1]:]
            print('3333')
            # 解码文本
            output_text = processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0].replace('If you have any other questions about the scene or anything else, feel free to ask.','')
            # # 音频转为 WAV bytes 并 base64 编码
            # buffer = io.BytesIO()
            # sf.write(buffer, audio_output[0].cpu().numpy(), samplerate=24000, format="WAV")
            # buffer.seek(0)
            # audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

            future.set_result({"text": output_text})
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()

app = FastAPI(title="Qwen2.5-Omni API", version="1.0")

class InferenceRequest(BaseModel):
    conversation: List[Dict[str, Any]]
    use_audio_in_video: bool = True

class InferenceResponse(BaseModel):
    text: str

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(worker())

@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    await request_queue.put({"data": request.dict(), "future": future})

    try:
        result = await future
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("omni_host:app", host="0.0.0.0", port=8081) 