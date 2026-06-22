import torch


def get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def get_model_primary_device(model) -> torch.device:
    return next(model.parameters()).device


def get_module_device(model, module_or_name) -> torch.device:
    module = module_or_name if not isinstance(module_or_name, str) else model.get_submodule(module_or_name)
    try:
        return next(module.parameters()).device
    except StopIteration:
        for _, buf in module.named_buffers(recurse=False):
            return buf.device
        return get_model_primary_device(model)


def is_sharded_device_spec(device: str) -> bool:
    if device is None:
        return False
    dev = device.strip().lower()
    return dev == 'auto' or ',' in dev
