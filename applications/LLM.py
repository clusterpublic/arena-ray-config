
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve import Application

from upload_file import upload_to_s3
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline,DiffusionPipeline 
from typing import Dict

from vllm import LLM,SamplingParams
import time
app = FastAPI()
from pydantic import BaseModel

class RequestBody(BaseModel):
    prompt: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 16
    min_tokens:  int = 0


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
        generated_text = await self.handle.generate.remote(body)
        end_time = time.time()
        resp = {
            "text":generated_text,
            "time_taken":end_time-start_time,
            "prompt":body.prompt
        }
        return JSONResponse(content=resp)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class LLMApplication:
    def __init__(self,model_id:str):
        print("n\n","model_id",model_id,"\n\n")
        self.model = LLM(model=model_id)
    def generate(self, body:RequestBody):
        prompts = []
        prompts.append(body.prompt)
        sampling_params=SamplingParams(temperature=body.temperature,top_p=body.top_p,min_tokens=body.min_tokens,max_tokens=body.max_tokens)
        output = self.model.generate(prompts,sampling_params)
        text= output[0].outputs[0].text
        return text


def app_builder(args: Dict[str, str]) -> Application:
    return APIIngress.options(route_prefix=args["route_prefix"]).bind(LLMApplication.bind(args["model_id"]))


