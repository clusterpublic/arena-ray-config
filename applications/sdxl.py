
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve import Application

from upload_file import upload_to_s3,generate_random_string_id

from diffusers import AutoPipelineForText2Image
import torch
from typing import Dict

import time
import datetime
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
    num_inference_steps:float=25
    guidance_scale:float=7.5
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
        start_timestamp = datetime.datetime.now().isoformat()
        start_time = time.time()
        model_ouput = await self.handle.generate.remote(body)
        upload_response= upload_to_s3(file_name=model_ouput["file_name"],object_name=model_ouput["file_name"],content_type="image/png")
        end_time = time.time()
        completion_timestamp = datetime.datetime.now().isoformat()
        resp = {
            "completed_at": completion_timestamp,
            "created_at": start_timestamp,
            "error": None,
            "input":body.dict(),
            "metrics": {
                "total_time": end_time-start_time,
            },
            "output": [
                    upload_response["url"]
            ],
            "started_at": start_timestamp,
            "status": "succeeded"
                    
                }
        return JSONResponse(content=resp)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class SDApplication:
    def __init__(self,model_id:str):
        print("n\n","model_id",model_id,"\n\n")
        
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
    
        
    def generate(self, body:RequestBody):

        image = self.pipe(prompt=body.prompt, num_inference_steps=body.num_inference_steps, guidance_scale=body.guidance_scale).images[0]


        file_name=f"output{generate_random_string_id()}.png"
        image.save(file_name)
        return {
            "success":True,
            "file_name":file_name
        }

def app_builder(args: Dict[str, str]) -> Application:
    return APIIngress.options(route_prefix=args["route_prefix"]).bind(SDApplication.bind(args["model_id"]))


