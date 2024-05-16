# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest

from optimum.intel import OVModelForCausalLM
from pathlib import Path
from py_continuous_batching import ContinuousBatchingPipeline, GenerationConfig, SchedulerConfig, GenerationResult
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from typing import List, Tuple
from openvino._pyopenvino.op import _PagedAttentionExtension


def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    return generation_config


def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_groups = 3
    generation_config.group_size = 2
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = generation_config.num_groups * generation_config.group_size
    return generation_config


def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada"
    ]
    generation_configs = [
        get_greedy(),
        get_beam_search(),
        get_greedy(),
        get_beam_search()
    ]
    return (prompts, generation_configs)


def get_scheduler_config(scheduler_params: dict = None) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    if scheduler_params is None:
        scheduler_config.dynamic_split_fuse = True
        # vLLM specific
        scheduler_config.max_num_batched_tokens = 256
        scheduler_config.max_num_seqs = 256
    else:
        for param, value in scheduler_params.items():
            setattr(scheduler_config, param, value)

    return scheduler_config


def convert_to_hf(
    default_generation_config : HFGenerationConfig,
    generation_config : GenerationConfig
) -> HFGenerationConfig:
    kwargs = {}

    # generic parameters
    kwargs['max_length'] = generation_config.max_length
    # has higher priority than 'max_length'
    kwargs['max_new_tokens'] = generation_config.max_new_tokens

    # copy default parameters
    kwargs['eos_token_id'] = default_generation_config.eos_token_id
    kwargs['pad_token_id'] = default_generation_config.pad_token_id

    if generation_config.num_groups * generation_config.group_size > 1:
        # beam search case
        kwargs['num_beam_groups'] = generation_config.num_groups
        kwargs['num_beams'] = generation_config.num_groups * generation_config.group_size
        kwargs['diversity_penalty'] = generation_config.diversity_penalty
        kwargs['repetition_penalty'] = generation_config.repetition_penalty
        kwargs['length_penalty'] = generation_config.length_penalty
        kwargs['no_repeat_ngram_size'] = generation_config.no_repeat_ngram_size
        kwargs['num_return_sequences'] = generation_config.num_return_sequences
        kwargs['output_scores'] = True
    elif generation_config.do_sample:
        # mulitinomial
        kwargs['temperature'] = generation_config.temperature
        kwargs['top_k'] = generation_config.top_k
        kwargs['top_p'] = generation_config.top_p
        kwargs['do_sample'] = generation_config.do_sample
    else:
        # greedy
        pass

    hf_generation_config = HFGenerationConfig(**kwargs)
    return hf_generation_config


def run_hugging_face(
    model_id : str,
    prompts: List[str],
    generation_configs: List[GenerationConfig],
    use_optimum: bool,
    tmp_path: Path
) -> Tuple[List[GenerationResult], str]:
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True) if use_optimum else \
            AutoModelForCausalLM.from_pretrained(model_id)
    generation_results: List[GenerationResult] = []
    model_path : Path = tmp_path / model_id

    if use_optimum:
        model.save_pretrained(model_path)
        # convert tokenizers as well
        from openvino_tokenizers import convert_tokenizer
        from openvino import serialize
        tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
        serialize(tokenizer, model_path / "openvino_tokenizer.xml")
        serialize(detokenizer, model_path / "openvino_detokenizer.xml")

    for prompt, generation_config in zip(prompts, generation_configs):
        inputs = hf_tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs['input_ids'].numel()
        generate_outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], generation_config=convert_to_hf(model.generation_config, generation_config), return_dict_in_generate=True)
        all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences])

        generation_result = GenerationResult()
        generation_result.m_generation_ids = all_text_batch
        # sequences_scores are available only for beam search case
        if generation_config.is_beam_search:
            generation_result.m_scores = [score for score in generate_outputs.sequences_scores]
        generation_results.append(generation_result)

    del hf_tokenizer
    del model

    return (generation_results, model_path)


def run_continuous_batching(
    model_path : Path,
    scheduler_config : SchedulerConfig,
    prompts: List[str],
    generation_configs : List[GenerationConfig]
) -> List[GenerationResult]:
    pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), scheduler_config)
    output = pipe.generate(prompts, generation_configs)
    del pipe
    shutil.rmtree(model_path)
    return output


def get_models_list(file_name: str):
    models = []
    with open(file_name) as f:
        for model_name in f:
            model_name = model_name.strip()
            # skip comment in model scope file
            if model_name.startswith('#'):
                continue
            models.append(model_name)
    return models


def compare_results(hf_result, ov_result, generation_config):
    if generation_config.is_beam_search:
        assert len(hf_result.m_scores) == len(ov_result.m_scores)
        for hf_score, ov_score in zip(hf_result.m_scores, ov_result.m_scores):
            # Note, that for fp32 / fp16 models scores are different less than 0.001
            assert abs(hf_score - ov_score) < 0.02

    assert len(hf_result.m_generation_ids) == len(ov_result.m_generation_ids)
    for hf_text, ov_text in zip(hf_result.m_generation_ids, ov_result.m_generation_ids):
        assert hf_text == ov_text


def run_test_pipeline(tmp_path: str, model_id: str, scheduler_params: dict = None):
    prompts, generation_configs = get_test_dataset()
    scheduler_config = get_scheduler_config(scheduler_params)

    (hf_results, model_path) = run_hugging_face(model_id=model_id, prompts=prompts,
                                                generation_configs=generation_configs, tmp_path=tmp_path,
                                                use_optimum=True)
    ov_results: List[GenerationResult] = run_continuous_batching(model_path, scheduler_config, prompts,
                                                                 generation_configs)

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(ov_results)

    for prompt, hf_result, ov_result, generation_config in zip(prompts, hf_results, ov_results, generation_configs):
        print(f"Prompt = {prompt}\nHF result = {hf_result}\nOV result = {ov_result}")
        compare_results(hf_result, ov_result, generation_config)

def run_pa(tmp_path, model_id):
    prompts, generation_configs = get_test_dataset()
    scheduler_config = get_scheduler_config(None)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForCausalLM.from_pretrained(model_id, export=True)
    model_path : Path = tmp_path / model_id
    model.save_pretrained(model_path)

    from openvino_tokenizers import convert_tokenizer
    from openvino import serialize
    tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    serialize(tokenizer, model_path / "openvino_tokenizer.xml")
    serialize(detokenizer, model_path / "openvino_detokenizer.xml")

    pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), scheduler_config)
    pipe.generate(prompts, generation_configs)

    ov_model = pipe.get_model()
    assert any(isinstance(op, _PagedAttentionExtension) for op in ov_model.get_ordered_ops())
