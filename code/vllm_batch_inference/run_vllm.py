import os
import json
import re
import jsonlines

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import argparse


os.environ["HF_HOME"] = "" # your cache location
os.environ["HF_TOKEN"] = "" # your hf token


def read_multiline_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Fix the JSON by wrapping in a list and replacing bad separators
    fixed_json = "[" + re.sub(r"}\s*{", "},{", raw_text.strip()) + "]"

    data = json.loads(fixed_json, strict=False)
    return data


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        data = [obj for obj in f]
    return data


def fix_tr_en_order(my_dict):
    tr = my_dict.pop("tr")
    en = my_dict.pop("en")

    my_dict["tr"] = tr
    my_dict["en"] = en
    return my_dict


def dataset_to_messages(dataset, system_tr, system_en, tokenizer):
    text_list = []

    for elem in dataset:
        for key in elem:
            if not key.endswith("_answer") and (
                key.startswith("tr") or key.startswith("en")
            ):
                message = [
                    {"role": "user", "content": elem[key]},
                ]

                message_tokenized = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )

                text_list.append(message_tokenized)

    return text_list


def return_bathces(outputs, size):
    for i in range(len(outputs) // size):
        yield outputs[size * i : size * i + size]


def run_experiment(
    dataset_path,
    output_root,
    llm,
    tokenizer,
    sampling_params,
    system_prompt_tr,
    system_prompt_en,
    model_name,
    idx_to_key,
    repeat=5,
):
    print(f"Start experiment for {dataset_path} and {model_name}")

    model_name = model_name[model_name.find("/") + 1 :]
    output_path = (
        dataset_path.replace(".jsonl", "") + "_" + model_name + "_L4" + ".jsonl"
    )

    if os.path.isfile(output_path):
        print(f"Output for {dataset_path} already exists, skipping")
        return

    try:
        data = read_jsonl(dataset_path)
    except Exception as e:
        data = read_multiline_json(dataset_path)

    print(f"Data head : {data[:2]}")

    # remove unused data
    for elem in data:
        if "tr_answer" in elem:
            elem.pop("tr_answer")
        if "en_answer" in elem:
            elem.pop("en_answer")

    text_list = dataset_to_messages(data, system_prompt_tr, system_prompt_en, tokenizer)
    print("Prepared prompt list")

    if "jailbreak" in dataset_path:
        print("Working on jailbreak")
        if sampling_params:
            outputs = llm.generate(text_list, sampling_params)
        else:
            outputs = llm.generate(text_list)
        for elem, answers in zip(data, return_bathces(outputs, 74)):
            elem[0] = {}
            for idx, answer in enumerate(answers):
                elem[0][f"{idx_to_key[idx]}_answer"] = answer.outputs[0].text
    else:
        print("Working on reliability, safety, stereorype")
        if sampling_params:
            outputs = llm.generate(text_list * 5, sampling_params)
        else:
            outputs = llm.generate(text_list * 5)

        batch_size = len(text_list)
        split_outputs = [
            outputs[i * batch_size : (i + 1) * batch_size] for i in range(repeat)
        ]

        for batch_index, batch in enumerate(split_outputs):
            for elem, (tr_answer, en_answer) in zip(data, return_bathces(batch, 2)):
                if batch_index not in elem:
                    elem[batch_index] = {}

                elem[batch_index]["tr_answer"] = tr_answer.outputs[0].text
                elem[batch_index]["en_answer"] = en_answer.outputs[0].text

    torch.cuda.empty_cache()
    print("Running finished")

    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(data)


def main(args):
    model = args.model
    tensor_parallel_size = args.tensor_parallel_size
    sampling_params = args.sampling_params
    max_tokens = args.max_tokens
    max_model_len = args.max_model_len
    quantization = args.quantization
    pipeline_parallel_size = args.pipeline_parallel_size

    datasets = [
        os.path.join("tr-toxic/data", f)
        for f in os.listdir("tr-toxic/data")
        if f.endswith(".jsonl")
    ]
    datasets = [dataset for dataset in datasets if "Qwen" not in dataset]
    datasets = [dataset for dataset in datasets if "Instruct" not in dataset]
    datasets = [dataset for dataset in datasets if "Llama" not in dataset]
    datasets = [dataset for dataset in datasets if "aya" not in dataset]

    datasets = [dataset for dataset in datasets if "OLMo" not in dataset]
    datasets = [dataset for dataset in datasets if "granite" not in dataset]
    
    datasets = [dataset for dataset in datasets if "gemma" not in dataset]
    datasets = [dataset for dataset in datasets if "Phi" not in dataset]
    
    datasets = [dataset for dataset in datasets if "phi" not in dataset]
    datasets = [dataset for dataset in datasets if "QwQ" not in dataset]
    datasets_dict = {}

    for dataset_path in datasets:
        try:
            dataset = read_jsonl(dataset_path)
        except Exception as e:
            dataset = read_multiline_json(dataset_path)

        dataset = [fix_tr_en_order(subdict) for subdict in dataset]
        datasets_dict[dataset_path] = dataset

    # Create adversarial dataset

    dataset_assert = [
        "tr-toxic/data/safety_data.jsonl",
        "tr-toxic/data/reliability_data.jsonl",
        "tr-toxic/data/stereotype_data.jsonl",
        "tr-toxic/data/jailbreak_data_fixed.jsonl",
    ]

    for key in datasets_dict:
        assert key in dataset_assert, f"Key not in dataset - key : {key}"

    idx_to_key = {}
    count = 0
    for key in datasets_dict["tr-toxic/data/jailbreak_data_fixed.jsonl"][0]:
        if not key.endswith("_answer") and (
            key.startswith("tr") or key.startswith("en")
        ):
            idx_to_key[count] = key
            count = count + 1

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90, # 0.9 works fine, higher causes OOM
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enforce_eager=True, # enable this for small models
        limit_mm_per_prompt={"image": 0}, # disable image
        enable_chunked_prefill=True,
        max_num_seqs=64, # low number to prevent OOM on vLLM v1 engine
    )
    
    if 'google' in model:
        sampling_params = SamplingParams(temperature = 1.0, top_k = 64, top_p = 0.95, min_p = 0.0, max_tokens=max_tokens) # gemma 3 recommended
    else:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_tokens) # qwen recommended

    for dataset in datasets:
        run_experiment(
            dataset_path=dataset,
            output_root="",
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            system_prompt_tr="",
            system_prompt_en="",
            model_name=model,
            idx_to_key=idx_to_key,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference with vLLM.")

    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--tensor_parallel_size", type=int, required=True, help="Tensor parallel size"
    )
    parser.add_argument("--pipeline_parallel_size", type=int, required=False, default=1)
    parser.add_argument(
        "--sampling_params",
        type=str,
        required=True,
        help="Sampling parameters type, like qwen, llama, aya etc.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="Max number of tokens model can emit per request",
    )
    parser.add_argument(
        "--max_model_len", type=int, required=True, help="Context window of model"
    )j
    parser.add_argument("--quantization", type=str, required=False, help="Quantization")
    args = parser.parse_args()

    main(args)