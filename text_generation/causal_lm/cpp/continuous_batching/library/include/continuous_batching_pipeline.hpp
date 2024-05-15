// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "scheduler_config.hpp"
#include "tokenizer.hpp"
#include "generation_config.hpp"

struct GenerationResult {
    // request ID
    uint64_t m_request_id;

    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<std::string> m_generation_ids;
    // scores
    std::vector<float> m_scores;
};

class ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config);

    std::shared_ptr<Tokenizer> get_tokenizer();

    GenerationConfig get_config() const;

    std::shared_ptr<ov::Model> get_model();

    void add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params);

    std::vector<GenerationResult> step();

    bool has_running_requests() const;

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params);
};
