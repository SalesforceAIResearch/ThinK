from importlib.metadata import version
import warnings
import transformers

from src.llama_model import llama_attn_forward_H2O, llama_attn_forward_SnapKV
from src.llama_model import llama_model_forward
from src.llama_model import prepare_inputs_for_generation_llama


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def replace_llama(method):
    transformers_version = check_version()
    version_list = ['4.40']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible.")
    
   
    transformers.models.llama.modeling_llama.LlamaModel.forward= llama_model_forward
    
        
    if method == "h2o":
        print("Using H2O!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        
        
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
