import torch

def modify_checkpoint(input_ckpt_path, output_ckpt_path, target_shapes):
    """
    Modify the checkpoint to match target shapes.

    Args:
        input_ckpt_path (str): Path to the input checkpoint file.
        output_ckpt_path (str): Path to save the modified checkpoint.
        target_shapes (dict): Dictionary specifying the target shapes for specific keys. 
                              Format: {key_name: (shape_tuple)}
    """
    # Load the checkpoint
    checkpoint = torch.load(input_ckpt_path, map_location=torch.device('cpu'))
    
    modified = False
    for key, param in checkpoint['model'].items():
        if key in target_shapes:
            target_shape = target_shapes[key]
            if param.shape != target_shape:
                print(f"Modifying {key}: {param.shape} -> {target_shape}")
                # Resize the parameter using slicing or padding
                resized_param = torch.zeros(target_shape)
                slices = tuple(slice(0, min(dim, target_dim)) for dim, target_dim in zip(param.shape, target_shape))
                resized_param[slices] = param[slices]
                checkpoint['model'][key] = resized_param
                modified = True

    if modified:
        # Save the modified checkpoint
        torch.save(checkpoint, output_ckpt_path)
        print(f"Modified checkpoint saved to {output_ckpt_path}")
    else:
        print("No modifications made. Target shapes already match.")

if __name__ == "__main__":
    # Input and output paths
    input_ckpt = "examples/bart/base/model.pt"
    output_ckpt = "examples/bart/base/model_new.pt"

    # Define target shapes for specific keys
    # Replace with your actual target shapes
    target_shapes = {
        "encoder.embed_tokens.weight": (50265, 768),
        "decoder.embed_tokens.weight": (50265, 768),
        "decoder.output_projection.weight": (50265, 768),
    }

    modify_checkpoint(input_ckpt, output_ckpt, target_shapes)
