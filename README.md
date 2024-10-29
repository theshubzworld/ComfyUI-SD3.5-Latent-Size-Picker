
# 🔧 SD3.5 Empty Latent Size Picker

A utility node for generating empty latent tensors in Stable Diffusion v3.5-compatible resolutions. This node allows for custom batch sizes, width/height overrides, and inverting aspect ratios, ensuring flexibility and compatibility in ComfyUI workflows.

## Features

- **Supported Resolutions**: Choose from a predefined list of Stable Diffusion v3.5-compatible resolutions.
- **Batch Size Control**: Set a batch size to create multiple latent tensors at once.
- **Width and Height Overrides**: Use custom width and height values as needed, while maintaining Stable Diffusion compatibility constraints.
- **Ratio Inversion Option**: Quickly swap width and height to experiment with alternate aspect ratios.
- **Multiple of 64 Constraint**: Ensures output width and height are multiples of 64, which Stable Diffusion requires for optimal performance.

## Parameters

- **resolution**: A dropdown list of pre-configured resolutions for Stable Diffusion v3.5, with a default of `"1024x1024 (1.0)"`.
- **batch_size**: Integer value defining the number of latent tensors to generate. Default is `1`.
- **width_override** and **height_override**: Optional custom dimensions. When set above 0, these values override the selected `resolution` dimensions.
- **invert_ratios**: Option to invert width and height for alternative aspect ratios.

## Output

The node returns:
- **LATENT**: A tensor with dimensions `[batch_size, 4, height // 8, width // 8]`, representing empty latent samples ready for further processing.
- **width**: Effective width, accounting for overrides and constraints.
- **height**: Effective height, accounting for overrides and constraints.

## Example Code

Here’s an example of how to use this node programmatically:

```python
import torch
from nodes.sd3_5_empty_latent import SD3_5EmptyLatent

# Initialize the SD3_5EmptyLatent node
latent_picker = SD3_5EmptyLatent()

# Set parameters
resolution = "1024x1024 (1.0)"
batch_size = 2
width_override = 768
height_override = 0
invert_ratios = "No"

# Execute to generate the latent tensor
latent_output, width, height = latent_picker.execute(
    resolution=resolution,
    batch_size=batch_size,
    width_override=width_override,
    height_override=height_override,
    invert_ratios=invert_ratios
)

print(f"Latent Shape: {latent_output['samples'].shape}, Width: {width}, Height: {height}")
```

## Installation

To use this node in ComfyUI:
1. Clone or download this repository.
2. Place the `SD3_5EmptyLatent` class script in the appropriate `nodes` folder for ComfyUI.
3. Restart ComfyUI to enable the node in the interface.

## License

This code is released under the MIT License, allowing for free use, distribution, and modification. See the `LICENSE` file for more details.
