import torch
import comfy.model_management
from nodes import MAX_RESOLUTION

class SD3_5EmptyLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "resolution": ([  # List of resolutions
                "640x1536 (0.98)", 
                "704x1344 (0.94)", 
                "768x1280 (0.98)", 
                "832x1216 (0.96)", 
                "896x1152 (1.03)", 
                "960x1024 (0.98)", 
                "1024x1024 (1.0)", 
                "1152x896 (1.03)", 
                "1216x832 (1.01)", 
                "1280x768 (0.98)",
                "1344x704 (0.95)"
            ], {"default": "1024x1024 (1.0)"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "height_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "invert_ratios": (["No", "Yes"], {"default": "No"}),  # New option for inverting ratios
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "sd3.5/utilities"

    def execute(self, resolution, batch_size, width_override=0, height_override=0, invert_ratios="No"):
        width, height = resolution.split(" ")[0].split("x")
        width = width_override if width_override > 0 else int(width)
        height = height_override if height_override > 0 else int(height)

        # If inverting ratios, swap width and height
        if invert_ratios == "Yes":
            width, height = height, width

        # Ensure that the width and height are multiples of 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples": latent}, width, height,) 

MISC_CLASS_MAPPINGS = {
    "SD3_5EmptyLatent": SD3_5EmptyLatent,
}

MISC_NAME_MAPPINGS = {
    "SD3_5EmptyLatent": "🔧 SD3.5 Empty Latent Size Picker",
}
