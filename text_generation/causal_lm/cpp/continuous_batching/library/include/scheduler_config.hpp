// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in constrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    std::size_t max_num_batched_tokens = 16;

    // TODO: specify size in GBs instead of number of KV blocks
    // total number of KV blocks available to scheduler logic
    std::size_t num_kv_blocks = 500;

    // block size for KV cache
    std::size_t block_size = 32;

    // whether to split prompt / generate to different scheduling phases
    bool dynamic_split_fuse = true;

    //
    // vLLM-like settings
    //

    // max number of scheduled sequences (you can think of it as "max batch size")
    std::size_t max_num_seqs = 256;
    // max number of padding tokens applied when we schedule a prompt phase
    // e.g. if total number of padded tokens within a batch a greater than this value, then
    // new sequnce is not added to batch
    std::size_t max_paddings = 256;
};