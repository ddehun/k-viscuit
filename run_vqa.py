from tqdm import tqdm
from datasets import load_dataset

from utils import set_seed, load_llava_model_and_processor_and_prompt


"""
Configurations
"""
set_seed(42)
hf_path = "ddehun/k-viscuit"
examples = load_dataset(hf_path)["test"]
model, processor, prompt_template = load_llava_model_and_processor_and_prompt()


"""
Run evaluation
"""
output_list = []
answer_list = []
for example in tqdm(examples):
    question, options, answer, image = example["question"], example["options"], example["answer"], example["image"]
    answer_list.append(answer)
    prompt = prompt_template.format(
        question=question, option_a=options[0], option_b=options[1], option_c=options[2], option_d=options[3]
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=3)
    generation = processor.decode(output[0], skip_special_tokens=True).strip()
    assert len(generation.split("[/INST]")) == 2, f"Error in {example['id_']}"
    generation = generation.split("[/INST]")[1].strip()
    output_list.append(generation)

"""
Post-process model generation and get accuracy
"""
assert len(output_list) == len(answer_list)
for i, e in enumerate(output_list):
    if e not in ["A", "B", "C", "D"]:
        output_list[i] = "-1"
        print("Error in", i, e)        
    else:
        output_list[i] = ord(e) - ord("A")

print("Acc (%): ", sum([a == b for a, b in zip(answer_list, output_list)]) / len(answer_list) * 100)
