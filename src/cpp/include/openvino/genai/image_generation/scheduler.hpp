// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>

#include "openvino/runtime/properties.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

/**
 * Scheduler used in image generation pipelines.
 */
class OPENVINO_GENAI_EXPORTS Scheduler {
public:
    /**
     * Defines scheduler type
     */
    enum Type {
        AUTO, ///< Scheduler type is automatically guessed from scheduler_config.json file
        LCM, ///< LCM scheduler
        LMS_DISCRETE, ///< LMS discrete scheduler
        DDIM, ///< DDIM scheduler
        EULER_DISCRETE, ///< Euler Discrete scheduler
        FLOW_MATCH_EULER_DISCRETE ///< Flow Match Euler Discrete scheduler
    };

    /**
     * Factory function to create scheduler based on the scheduler_config.json file.
     * @param scheduler_config_path Full path to scheduler_config.json file
     * @param scheduler_type Optional type of scheduler, by default it is auto-guessed based on scheduler_config.json
     */
    static std::shared_ptr<Scheduler> from_config(const std::filesystem::path& scheduler_config_path,
                                                  Type scheduler_type = AUTO);

    /**
     * Default Scheduler dtor used for RTTI
     */
    virtual ~Scheduler();
};

/**
 * Overrides default scheduler used in pipeline.
 * This is useful when OpenVINO GenAI does not support specific scheduler type and it is
 * recommended to manually create `Scheduler` instance and pass it to image generation pipeline constructor
 * or compile method as a property
 * 
 * To pass scheduler via pipeline constructor:
 * @code
 * auto scheduler = ov::genai::Scheduler::from_config(models_path / "scheduler/scheduler_config.json");
 * ov::genai::Text2ImagePipeline pipe(models_path, "CPU", ov::genai::scheduler(scheduler));
 * @endcode
 * 
 * Via `compile` method:
 * @code
 * auto scheduler = ov::genai::Scheduler::from_config(models_path / "scheduler/scheduler_config.json");
 * ov::genai::Image2ImagePipeline pipe(models_path);
 * pipe.compile("GPU", ov::genai::scheduler(scheduler));
 * @endcode
 */
static constexpr ov::Property<std::shared_ptr<Scheduler>> scheduler{"scheduler"};

} // namespace genai
} // namespace ov
