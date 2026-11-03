from llava.model.llava_arch import LlavaMetaForCausalLM

from method_wrappers.binding import bind_llava_methods

from .llava_arch import LlavaMetaForCausalLM_tempo
from .llava_arch_v1 import LlavaMetaForCausalLM_tempo_v1


def tempo(model):
    """Apply TeMPO (vectorized implementation) to a LLaVA-family model in place."""
    bind_llava_methods(LlavaMetaForCausalLM, LlavaMetaForCausalLM_tempo)
    return model


def tempo_v1(model):
    """Apply the loop-form TeMPO reference implementation (slower, readable)."""
    bind_llava_methods(LlavaMetaForCausalLM, LlavaMetaForCausalLM_tempo_v1)
    return model
