import torch

@torch.no_grad()
def add_feature_on_text_prompt(sae, steering_feature, module, input, output):
    """Add steering feature to text encoder output (conditional pass)."""
    # 1. parse unpack
    orig_output = output[0] if isinstance(output, tuple) else output
    orig_input = input[0] if isinstance(input, tuple) else input

    # 2. perform addition logic
    if orig_input.size(-1) == 768:
        modified = orig_output + steering_feature[:, :768].unsqueeze(0)
    else:
        modified = orig_output + steering_feature[:, 768:].unsqueeze(0)

    # 3. repack return, no longer use hardcoded
    if isinstance(output, tuple):
        return (modified,) + output[1:]
    return modified

@torch.no_grad()
def minus_feature_on_text_prompt(sae, steering_feature, module, input, output):
    """Subtract steering feature from text encoder output (unconditional pass)."""
    # 1. parse unpack
    orig_output = output[0] if isinstance(output, tuple) else output
    orig_input = input[0] if isinstance(input, tuple) else input

    # 2. perform subtraction logic
    if orig_input.size(-1) == 768:
        modified = orig_output - steering_feature[:, :768].unsqueeze(0)
    else:
        modified = orig_output - steering_feature[:, 768:].unsqueeze(0)

    # 3. repack return, no longer use hardcoded
    if isinstance(output, tuple):
        return (modified,) + output[1:]
    return modified

@torch.no_grad()
def do_nothing(sae, steering_feature, module, input, output):
    """No-op hook that returns original output without modification."""
    return output
