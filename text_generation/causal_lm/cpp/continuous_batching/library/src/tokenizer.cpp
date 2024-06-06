
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <fstream>

#include <jinja2cpp/template.h>
#include <jinja2cpp/template_env.h>
#include "nlohmann/json.hpp"

#include "openvino/runtime/core.hpp"

#include "tokenizer.hpp"

class Tokenizer::Impl {
    const size_t TOKENIZER_BATCH_SIZE = 1;

    // Execution 
    ov::InferRequest m_tokenizer;
    ov::InferRequest m_detokenizer;

    // EOS token ID read from OV model
    std::size_t m_eos_token_id;

    // Configuration from tokenizer_config.json 
    std::string m_eos_token;
    std::string m_bos_token;
    std::string m_chat_template;

    // Synchronization. Using multiple infer requests hangs. For now we synchronize entire execution on a single infer request.
    std::mutex m_tokenizer_mutex;
    std::mutex m_detokenizer_mutex;

    // Chat template handling
    std::unique_ptr<jinja2::TemplateEnv> m_template_env;
    std::unique_ptr<jinja2::Template> m_processed_chat_template;

public:
    explicit Impl(const std::string& models_path)
    {
        ov::Core core;
        core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

        std::shared_ptr<ov::Model> tokenizer_model = core.read_model(models_path + "/openvino_tokenizer.xml");
        const ov::AnyMap& rt_info = tokenizer_model->get_rt_info();
        OPENVINO_ASSERT(rt_info.find("eos_token_id") != rt_info.end(), "Failed to detect \"eos_token_id\" in openvino_tokenizer.xml runtime information");
        m_eos_token_id = rt_info.at("eos_token_id").as<int64_t>();

        // tokenizer and detokenizer work on CPU only
        m_tokenizer = core.compile_model(
            tokenizer_model, "CPU").create_infer_request();
        m_detokenizer = core.compile_model(
            models_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();

        std::ifstream tokenizer_config(models_path + "/tokenizer_config.json");
        nlohmann::json json_data = nlohmann::json::parse(tokenizer_config);

        std::cout << "Read tokenizer config" << std::endl;
        m_bos_token = json_data.value("bos_token", "");
        m_eos_token = json_data.value("eos_token", "");
        m_chat_template = json_data.value("chat_template", "");

        m_template_env = std::make_unique<jinja2::TemplateEnv>();
        m_template_env->GetSettings().lstripBlocks = true;
        m_template_env->GetSettings().trimBlocks = true;
        m_processed_chat_template = std::make_unique<jinja2::Template>(m_template_env.get());
        m_processed_chat_template->Load(m_chat_template);
    }

    ov::Tensor encode(std::string prompt) {
        std::unique_lock<std::mutex> lock(m_tokenizer_mutex);
        m_tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {TOKENIZER_BATCH_SIZE}, &prompt});
        m_tokenizer.infer();
        ov::Tensor tmp_tensor = m_tokenizer.get_tensor("input_ids");
        ov::Tensor output_tensor(tmp_tensor.get_element_type(), tmp_tensor.get_shape());
        tmp_tensor.copy_to(output_tensor);
        return output_tensor;
    }

    std::string decode(std::vector<int64_t> tokens) {
        std::unique_lock<std::mutex> lock(m_detokenizer_mutex);
        m_detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {TOKENIZER_BATCH_SIZE, tokens.size()}, tokens.data()});
        m_detokenizer.infer();
        return m_detokenizer.get_output_tensor().data<std::string>()[0];
    }

    size_t get_eos_token_id() const {
        return m_eos_token_id;
    }

    std::string apply_chat_template(chat_t chat) {
        jinja2::ValuesList valuesList;
        for (auto& m : chat) {
            std::string role = m["role"];
            std::string prompt = m["content"];
            jinja2::ValuesMap message {{"role", role}, {"content", prompt}};
            valuesList.emplace_back(message);
        }
        jinja2::ValuesMap params = {
            {"messages", valuesList},
            {"bos_token",  m_bos_token},
            {"eos_token", m_eos_token},
            {"add_generation_prompt", true},
        };
        std::string text = m_processed_chat_template->RenderAsString(params).value();
        return text;
    }
};

Tokenizer::Tokenizer(const std::string& models_path) {
    m_impl = std::make_shared<Impl>(models_path);
}

ov::Tensor Tokenizer::encode(std::string prompt) {
    return m_impl->encode(prompt);
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_impl->decode(tokens);
}

size_t Tokenizer::get_eos_token_id() const {
    return m_impl->get_eos_token_id();
}

std::string Tokenizer::apply_chat_template(chat_t chat) {
    return m_impl->apply_chat_template(chat);
}
