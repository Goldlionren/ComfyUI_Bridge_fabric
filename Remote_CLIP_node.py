
import torch
import folder_paths
import os
import time

class RemoteCLIPHostSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "filename_prefix": ("STRING", {"default": "wan_remote_tensors"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_tensors"
    OUTPUT_NODE = True
    CATEGORY = "WanRemote"

    def save_tensors(self, positive, negative, filename_prefix):
        # We save to a specific known filename that the client can request
        # Adding timestamp to prevent caching issues, but returning the filename
        filename = f"{filename_prefix}.pt"
        file_path = os.path.join(self.output_dir, filename)
        
        print(f"--- WanRemote Host: Saving Tensors to {file_path} ---")
        
        # Save dictionary
        torch.save({
            "positive": positive,
            "negative": negative
        }, file_path)
        
        # Return ui update to let ComfyUI know we are done
        return {"ui": {"text": ["Tensors Saved"]}}

NODE_CLASS_MAPPINGS = {
    "RemoteCLIPHostSaver": RemoteCLIPHostSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteCLIPHostSaver": "Remote CLIP Host Saver"
}
