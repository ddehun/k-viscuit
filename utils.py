from typing import List, Dict, Any, Tuple
import os
import random

import numpy as np
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def load_llava_model_and_processor_and_prompt(
    name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor, str]:
    assert "mistral" in name and "llava" in name, name
    processor = LlavaNextProcessor.from_pretrained(name)
    model = LlavaNextForConditionalGeneration.from_pretrained(name, load_in_8bit=True).eval()
    
    prompt = """[INST] <image>\n{question}
    A. {option_a}
    B. {option_b}
    C. {option_c}
    D. {option_d}
    Answer with the option's letter from the given choices directly. [/INST]"""
    return model, processor, prompt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
