
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve import Application

from upload_file import upload_to_s3,generate_random_string_id

import torch
from typing import Dict
import scipy

from transformers import AutoProcessor, BarkModel
import time
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel

class RequestBody(BaseModel):
    prompt: str
    # temperature: float = 1.0
    # top_p: float = 1.0
    # max_tokens: int = 16
    # min_tokens:  int = 0


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.post(
        "/",
        responses={200: {"content": {}}},
        response_class=JSONResponse,
    )
    async def generate(self, body: RequestBody):
        start_time = time.time()
        model_ouput = await self.handle.generate.remote(body)
        end_time = time.time()
        completion_time= end_time-start_time
        upload_response= upload_to_s3(file_name=model_ouput["file_name"],object_name=model_ouput["file_name"],content_type="audio/wav")
        resp = {
            "image":{
                "url":upload_response["url"]
                },
            "completion_time":completion_time,
            "prompt":body.prompt
        }
        return JSONResponse(content=resp)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class SunoApplication:
    def __init__(self,model_id:str):
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    def generate(self, body: RequestBody, ):
        voice_preset = "v2/en_speaker_6"
        text_prompt = body.prompt
        inputs = self.processor(text_prompt, voice_preset=voice_preset, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze() # Move to CPU for further processing if needed
        sample_rate = self.model.generation_config.sample_rate
        file_name=f"output-{generate_random_string_id()}.wav"
        scipy.io.wavfile.write(file_name ,rate=sample_rate, data=audio_array)
        return {
            "success":True,
            "file_name":file_name
        }

def app_builder(args: Dict[str, str]) -> Application:
    return APIIngress.options(route_prefix=args["route_prefix"]).bind(SunoApplication.bind(args["model_id"]))


