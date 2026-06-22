"""
Step 1a: Verify hparams module-path templates actually resolve to the
intended weight tensors on the new architecture.

Run this BEFORE touching any editing code. If a path doesn't resolve,
or resolves to a module of unexpected shape, that's your bug located.
"""
from transformers import AutoModelForCausalLM

def check_paths(model_name, layer_idx, paths: dict):
    print(f"=== {model_name} (layer {layer_idx}) ===")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    named = dict(model.named_modules())
    for label, tmpl in paths.items():
        resolved = tmpl.format(layer_idx)
        mod = named.get(resolved, None)
        if mod is None:
            print(f"  [MISSING] {label}: '{resolved}' does NOT exist in model.named_modules()")
        else:
            shape = getattr(mod, "weight", None)
            shape = tuple(shape.shape) if shape is not None else "no .weight"
            print(f"  [OK]      {label}: '{resolved}' -> {type(mod).__name__}, weight shape={shape}")
    print()

# --- Reference: what AlphaEdit assumes for a Llama-style model ---
llama_style = {
    "rewrite_module (down_proj)": "model.layers.{}.mlp.down_proj",
    "mlp_module": "model.layers.{}.mlp",
    "attn_module": "model.layers.{}.self_attn",
    "layer_module": "model.layers.{}",
}

# --- Fill in what your hparams JSON currently specifies for each new model ---
gemma2_paths_to_test = {
    "rewrite_module (down_proj)": "model.layers.{}.mlp.down_proj",   # CONFIRM this exists
    "mlp_module": "model.layers.{}.mlp",
    "attn_module": "model.layers.{}.self_attn",
    "layer_module": "model.layers.{}",
}

phi3_paths_to_test = {
    "rewrite_module (down_proj)": "model.layers.{}.mlp.down_proj",   # Phi-3 uses gate_up_proj + down_proj
    "mlp_module": "model.layers.{}.mlp",
    "attn_module": "model.layers.{}.self_attn",
    "layer_module": "model.layers.{}",
}

if __name__ == "__main__":
    check_paths("google/gemma-2-2b", 5, gemma2_paths_to_test)
    check_paths("microsoft/Phi-3-mini-4k-instruct", 5, phi3_paths_to_test)

    # Also just dump real module names so you can see ground truth directly:
    for name in ["google/gemma-2-2b", "microsoft/Phi-3-mini-4k-instruct"]:
        print(f"=== Real module names containing 'mlp' or 'proj' in {name}, layer 5 ===")
        model = AutoModelForCausalLM.from_pretrained(name)
        for n, _ in model.named_modules():
            if "layers.5." in n and ("mlp" in n or "proj" in n):
                print(" ", n)
        print()
