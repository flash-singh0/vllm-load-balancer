# # app.py
# import runpod
# import subprocess
# import os
# import time
# import requests
# from typing import Optional

# class VLLMWorker:
#     def __init__(self):
#         self.port = int(os.getenv("PORT", "8000"))
#         self.host = "0.0.0.0"
#         self.model_name = os.getenv("MODEL_NAME", "facebook/opt-125m")
#         self.process: Optional[subprocess.Popen] = None
        
#     def check_ready(self) -> bool:
#         """Check if VLLM server is responsive"""
#         try:
#             response = requests.get(
#                 f"http://{self.host}:{self.port}/v1/models",
#                 timeout=5
#             )
#             return response.status_code == 200
#         except:
#             return False
            
#     def ensure_server(self) -> bool:
#         """Start VLLM server if not running and wait for it to be ready"""
#         # If process exists, check if it's still running
#         if self.process is not None:
#             if self.process.poll() is None and self.check_ready():
#                 return True
#             # Process is dead or unresponsive, kill it
#             try:
#                 self.process.kill()
#             except:
#                 pass
#             self.process = None
        
#         # Build command
#         cmd = [
#             "vllm",
#             "serve",
#             "--model", self.model_name,
#             "--host", self.host,
#             "--port", str(self.port),
#             "--trust-remote-code"
#         ]
        
#         # Start server
#         self.process = subprocess.Popen(cmd)
        
#         # Wait for server to become responsive
#         start_time = time.time()
#         timeout = 600  # 10 minute timeout
#         while time.time() - start_time < timeout:
#             if self.check_ready():
#                 return True
#             time.sleep(1)
        
#         return False
        
#     def generate(self, params: dict) -> dict:
#         """Send generation request to VLLM server"""
#         try:
#             response = requests.post(
#                 f"http://{self.host}:{self.port}/v1/completions",
#                 json={
#                     "prompt": params.get("prompt", ""),
#                     "max_tokens": params.get("max_tokens", 100),
#                     "temperature": params.get("temperature", 0.7),
#                     "top_p": params.get("top_p", 1.0),
#                     "top_k": params.get("top_k", -1),
#                     "stream": False
#                 },
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 return {
#                     "status_code": 200,
#                     "generated_text": response.json()["choices"][0]["text"]
#                 }
#             else:
#                 return {
#                     "error": f"VLLM server error: {response.text}",
#                     "status_code": response.status_code
#                 }
                
#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "status_code": 500
#             }

# def handler(job):
#     """RunPod handler function"""
#     # Create worker instance for this job
#     worker = VLLMWorker()
#     job_input = job["input"]
    
#     # Handle ping request
#     if job_input.get("ping"):
#         if not worker.check_ready():
#             return {"status_code": 204}
#         return {"status_code": 200}
    
#     # Ensure server is running for generation requests
#     if not worker.ensure_server():
#         return {
#             "error": "Failed to start VLLM server",
#             "status_code": 500
#         }
    
#     # Process generation request
#     return worker.generate(job_input)

# # Start the RunPod handler
# runpod.serverless.start({"handler": handler})

from fastapi import FastAPI, HTTPException
import subprocess
import os
import time
import requests
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1

class VLLMWorker:
    def __init__(self):
        self.port = int(os.getenv("PORT", "8000"))
        self.host = "0.0.0.0"
        self.model_name = os.getenv("MODEL_NAME", "facebook/opt-125m")
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False
        self.start_server()
        
    def check_ready(self) -> bool:
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/v1/models",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
            
    def start_server(self):
        if self.process is not None:
            try:
                self.process.kill()
            except:
                pass
            
        cmd = [
            "vllm",
            "serve",
            self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--trust-remote-code"
        ]
        
        self.process = subprocess.Popen(cmd)
        
        # Wait for initial startup
        start_time = time.time()
        timeout = 600  # 10 minute timeout
        while time.time() - start_time < timeout:
            if self.check_ready():
                self.initialized = True
                break
            time.sleep(1)
        
    def generate(self, params: GenerationRequest) -> dict:
        if not self.initialized:
            raise HTTPException(status_code=503, detail="Server still initializing")
            
        try:
            response = requests.post(
                f"http://{self.host}:{self.port}/v1/completions",
                json={
                    "prompt": params.prompt,
                    "max_tokens": params.max_tokens,
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "top_k": params.top_k,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    "generated_text": response.json()["choices"][0]["text"]
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"VLLM server error: {response.text}"
                )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Create global worker instance
worker = VLLMWorker()

@app.get("/ping")
async def health_check():
    if worker.process is None or worker.process.poll() is not None:
        raise HTTPException(status_code=500, detail="Server is unhealthy")
    if not worker.initialized:
        return {"status": "initializing"}
    if worker.check_ready():
        return {"status": "ready"}
    raise HTTPException(status_code=500, detail="Server is unhealthy")

@app.post("/generate")
async def generate(request: GenerationRequest):
    return worker.generate(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)