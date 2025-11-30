
import torch
import requests
import json
import time
import io
import random

class LocalCLIPClient:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "server_ip": ("STRING", {"default": "http://10.10.10.211:8188", "multiline": False}),
                "remote_clip_name": ("STRING", {"default": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "multiline": False}),
                "clip_type": ([
                    "wan",
                    "stable_diffusion",
                    "stable_cascade",
                    "sd3",
                    "stable_audio",
                    "hunyuan_dit",
                    "flux",
                    "mochi",
                    "ltxv",
                    "hunyuan_video",
                    "pixart",
                    "cosmos",
                    "lumina2",
                    "hidream",
                    "chroma",
                    "ace",
                    "omnigen2",
                    "qwen_image",
                    "hunyuan_image",
                    "hunyuan_video_15"
                ], {"default": "wan"}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "dispatch_remote"
    CATEGORY = "WanRemote"

    def dispatch_remote(self, server_ip, remote_clip_name, clip_type, positive_prompt, negative_prompt):
        # Cleanup URL
        base_url = server_ip.rstrip("/")
        if not base_url.startswith("http"):
            base_url = "http://" + base_url
            
        print(f"--- WanRemote Client: Connecting to {base_url} [Type: {clip_type}] ---")
        
        # 1. Construct the Remote Workflow JSON
        # This assumes the Remote Machine has standard "CLIPLoader", "CLIPTextEncode" and our "WanRemote_Host" node.
        
        # Random seed to force re-execution if needed, though CLIP is deterministic
        client_id = str(random.randint(0, 1000000))
        
        # Define Node IDs
        ID_LOADER = "10"
        ID_POS = "20"
        ID_NEG = "30"
        ID_SAVER = "40"
        
        # Wan2.1 uses T5, so we use standard CLIPLoader with type 'wan' or standard loading
        # NOTE: User must ensure the remote machine has the model in models/clip/
        
        workflow = {
            ID_LOADER: {
                "inputs": {
                    "clip_name": remote_clip_name,
                    "type": clip_type 
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Remote CLIP Loader"}
            },
            ID_POS: {
                "inputs": {
                    "text": positive_prompt,
                    "clip": [ID_LOADER, 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Pos Encode"}
            },
            ID_NEG: {
                "inputs": {
                    "text": negative_prompt,
                    "clip": [ID_LOADER, 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Neg Encode"}
            },
            ID_SAVER: {
                "inputs": {
                    "positive": [ID_POS, 0],
                    "negative": [ID_NEG, 0],
                    "filename_prefix": f"wan_remote_{client_id}"
                },
                "class_type": "RemoteCLIPHostSaver", # Must match the Host Node Class Name
                "_meta": {"title": "Remote Saver"}
            }
        }
        
        prompt_payload = {
            "prompt": workflow,
            "client_id": client_id
        }
        
        # 2. Send Prompt
        try:
            req = requests.post(f"{base_url}/prompt", json=prompt_payload)
            req.raise_for_status()
            response = req.json()
            prompt_id = response.get("prompt_id")
            print(f"   Workflow sent. Prompt ID: {prompt_id}")
        except Exception as e:
            print(f"   Error sending prompt: {e}")
            raise e
            
        # 3. Wait for Execution
        # Simple polling loop. (Ideally use WebSocket, but polling /history is easier for a simple script)
        print("   Waiting for remote execution...")
        start_time = time.time()
        timeout = 300 # 5 minutes
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Remote generation timed out.")
            
            time.sleep(1) # Poll every second
            
            try:
                hist_req = requests.get(f"{base_url}/history/{prompt_id}")
                if hist_req.status_code == 200:
                    hist = hist_req.json()
                    if prompt_id in hist:
                        # Execution finished
                        print("   Remote execution finished.")
                        break
            except:
                pass
                
        # 4. Download Result
        filename = f"wan_remote_{client_id}.pt"
        download_url = f"{base_url}/view?filename={filename}&type=output"
        
        print(f"   Downloading result from {download_url}...")
        
        try:
            r = requests.get(download_url)
            r.raise_for_status()
            
            # Load bytes into memory
            buffer = io.BytesIO(r.content)
            
            # 5. Load Tensor
            # map_location='cpu' is important to avoid CUDA device mismatch
            data = torch.load(buffer, map_location="cpu")
            
            pos_cond = data["positive"]
            neg_cond = data["negative"]
            
            print("   Tensors loaded successfully.")
            return (pos_cond, neg_cond)
            
        except Exception as e:
            print(f"   Error downloading/loading result: {e}")
            raise e

NODE_CLASS_MAPPINGS = {
    "LocalCLIPClient": LocalCLIPClient
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalCLIPClient": "Local CLIP Client"
}
