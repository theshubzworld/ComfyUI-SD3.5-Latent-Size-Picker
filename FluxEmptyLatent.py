import torch
import comfy.model_management
from nodes import MAX_RESOLUTION
from typing import Dict, Tuple

class FluxEmptyLatent:
    """
    An optimized empty latent node with extended aspect ratio support.
    Designed for Flux models but compatible with SD3.5 and SDXL.
    """
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": ([  # Extended range of resolutions
                    # Square
                    "512x512 (1:1)", "576x576 (1:1)", "640x640 (1:1)", "704x704 (1:1)",
                    "768x768 (1:1)", "832x832 (1:1)", "896x896 (1:1)", "960x960 (1:1)",
                    "1024x1024 (1:1)", "1088x1088 (1:1)", "1152x1152 (1:1)", "1216x1216 (1:1)",
                    "1280x1280 (1:1)", "1344x1344 (1:1)", "1408x1408 (1:1)", "1472x1472 (1:1)",
                    "1536x1536 (1:1)",
                    
                    # Portrait
                    "512x768 (2:3)", "576x864 (2:3)", "640x960 (2:3)", "704x1056 (2:3)",
                    "768x1152 (2:3)", "832x1248 (2:3)", "896x1344 (2:3)", "960x1440 (2:3)",
                    "1024x1536 (2:3)",
                    
                    # Landscape
                    "768x512 (3:2)", "864x576 (3:2)", "960x640 (3:2)", "1056x704 (3:2)",
                    "1152x768 (3:2)", "1248x832 (3:2)", "1344x896 (3:2)", "1440x960 (3:2)",
                    "1536x1024 (3:2)",
                    
                    # Ultra Portrait
                    "512x1024 (1:2)", "576x1152 (1:2)", "640x1280 (1:2)", "704x1408 (1:2)",
                    "768x1536 (1:2)",
                    
                    # Ultra Landscape
                    "1024x512 (2:1)", "1152x576 (2:1)", "1280x640 (2:1)", "1408x704 (2:1)",
                    "1536x768 (2:1)",
                    
                    # Mobile/Story
                    "512x896 (4:7)", "576x1008 (4:7)", "640x1120 (4:7)", "704x1232 (4:7)",
                    "768x1344 (4:7)", "832x1456 (4:7)",
                    
                    # Widescreen
                    "1024x576 (16:9)", "1152x648 (16:9)", "1280x720 (16:9)", "1408x792 (16:9)",
                    "1536x864 (16:9)", "1664x936 (16:9)", "1792x1008 (16:9)", "1920x1080 (16:9)",
                    
                    # Custom Ratios
                    "640x1536 (5:12)", "704x1408 (1:2)", "768x1280 (3:5)", "832x1216 (13:19)",
                    "896x1152 (7:9)", "960x1024 (15:16)", "1152x896 (9:7)", "1216x832 (19:13)",
                    "1280x768 (5:3)", "1344x704 (21:11)", "1408x640 (11:5)", "1536x576 (8:3)",
                    "1600x512 (25:8)"
                ], {"default": "1024x1024 (1:1)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "aspect_ratio_lock": (["Unlocked", "Width", "Height"], {"default": "Unlocked"}),
                "latent_channels": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "downsample_factor": (["auto", 4, 8], {"default": "auto"}),
                "invert_ratios": (["No", "Yes"], {"default": "No"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT",)
    RETURN_NAMES = ("LATENT", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "latent"
    
    def calculate_downsample_factor(self, downsample_setting: str) -> int:
        """Determine the downsample factor based on user setting."""
        if downsample_setting == "auto":
            # Default to 8 for better compatibility
            return 8
        return int(downsample_setting)

    def execute(
        self,
        resolution: str,
        batch_size: int,
        width_override: int = 0,
        height_override: int = 0,
        aspect_ratio_lock: str = "Unlocked",
        latent_channels: int = 4,
        downsample_factor: str = "auto",
        invert_ratios: str = "No"
    ) -> Tuple[Dict, int, int]:
        """
        Create an empty latent tensor with the specified dimensions.
        Optimized for performance with extended aspect ratio support.
        
        Args:
            resolution: The base resolution string (e.g., "1024x1024 (1:1)")
            batch_size: Number of latent tensors to create
            width_override: Override width if > 0
            height_override: Override height if > 0
            aspect_ratio_lock: Lock aspect ratio to "Width", "Height", or "Unlocked"
            latent_channels: Number of channels in latent space (4 for SD, 16 for Flux)
            downsample_factor: Downsample factor (4, 8, or "auto" for 8)
            invert_ratios: "Yes" to swap width and height
            
        Returns:
            Tuple containing (latent_dict, width, height)
        """
        # 1. Parse base resolution
        try:
            res_part = resolution.split(" ")[0]
            base_width, base_height = map(int, res_part.split("x"))
        except (ValueError, IndexError):
            raise ValueError(f"Invalid resolution format: {resolution}. Expected format: 'WIDTHxHEIGHT (RATIO)'")

        # 2. Apply overrides with aspect ratio lock
        width, height = base_width, base_height
        
        if aspect_ratio_lock == "Width" and width_override > 0:
            width = min(width_override, MAX_RESOLUTION)
            height = max(64, int(round((base_height / base_width) * width)))
        elif aspect_ratio_lock == "Height" and height_override > 0:
            height = min(height_override, MAX_RESOLUTION)
            width = max(64, int(round((base_width / base_height) * height)))
        else:
            if width_override > 0:
                width = min(width_override, MAX_RESOLUTION)
            if height_override > 0:
                height = min(height_override, MAX_RESOLUTION)
        
        # 3. Apply ratio inversion if needed
        if invert_ratios == "Yes":
            width, height = height, width

        # 4. Ensure valid dimensions
        width = max(64, min(width, MAX_RESOLUTION))
        height = max(64, min(height, MAX_RESOLUTION))
        
        # 5. Calculate latent dimensions
        factor = self.calculate_downsample_factor(downsample_factor)
        latent_width = (width + factor - 1) // factor
        latent_height = (height + factor - 1) // factor
        
        # 6. Create the latent tensor (optimized for performance)
        latent = torch.zeros(
            [batch_size, latent_channels, latent_height, latent_width],
            device=self.device,
            dtype=torch.float32
        )
        
        # Return the actual pixel dimensions that the latent represents
        actual_width = latent_width * factor
        actual_height = latent_height * factor
        
        return ({"samples": latent}, actual_width, actual_height)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxEmptyLatent": FluxEmptyLatent,
}

# Display names for the nodes in the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxEmptyLatent": "🔧 Flux Empty Latent (Enhanced)"
}