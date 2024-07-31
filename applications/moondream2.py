
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve import Application

from upload_file import generate_random_string_id
from typing import Dict
import time
import base64
import datetime
from PIL import Image

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
    image: str


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
        uploaded_image_data = base64.b64decode(body.image)
        uploaded_image = Image.open(BytesIO(uploaded_image_data))
        temp_file_name=f"{generate_random_string_id()}_uploaded_image.png"
        uploaded_image.save(temp_file_name)
        text_prompt = body.prompt
        print(f"Received text prompt: {text_prompt}")
        start_timestamp = datetime.datetime.now().isoformat()
        start_time = time.time()
        output_data = await self.handle.generate.remote(body.prompt,temp_file_name)
        end_time = time.time()
        completion_timestamp = datetime.datetime.now().isoformat()
        resp = {
            "completed_at": completion_timestamp,
            "created_at": start_timestamp,
            "error": None,
            "input":{
                "prompt":body.prompt
                },
            "metrics": {
                "total_time": end_time-start_time,
            },
            "output": [
                output_data["response"]
            ],
            "started_at": start_timestamp,
            "status": "succeeded"
                    
                }
        return JSONResponse(content=resp)


@serve.deployment(
    ray_actor_options={"num_gpus": 1,"num_cpus":2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class MoonDreamApplication:
    def __init__(self,model_id:str):
        print("n\n","model_id",model_id,"\n\n")
        
        model_id = "vikhyatk/moondream2"
        revision = "2024-07-23"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    def generate(self, prompt:str,image_path:str):
        
        image = Image.open(image_path)
        enc_image = self.model.encode_image(image)
        response= self.model.answer_question(enc_image, "Describe this image.", self.tokenizer)
        return {"success":True,"response":response}
        
def app_builder(args: Dict[str, str]) -> Application:
    return APIIngress.options(route_prefix=args["route_prefix"]).bind(MoonDreamApplication.bind(args["model_id"]))





