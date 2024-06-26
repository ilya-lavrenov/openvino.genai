// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "openvino/genai/llm_pipeline.hpp"

namespace py = pybind11;
using ov::genai::LLMPipeline;
using ov::genai::Tokenizer;
using ov::genai::GenerationConfig;
using ov::genai::EncodedResults;
using ov::genai::DecodedResults;
using ov::genai::StopCriteria;
using ov::genai::StreamerBase;
using ov::genai::StreamerVariant;
using ov::genai::OptionalGenerationConfig;

namespace {

void update_config_from_kwargs(GenerationConfig& config, const py::kwargs& kwargs) {
    if (kwargs.contains("max_new_tokens")) config.max_new_tokens = kwargs["max_new_tokens"].cast<size_t>();
    if (kwargs.contains("max_length")) config.max_length = kwargs["max_length"].cast<size_t>();
    if (kwargs.contains("ignore_eos")) config.ignore_eos = kwargs["ignore_eos"].cast<bool>();
    if (kwargs.contains("num_beam_groups")) config.num_beam_groups = kwargs["num_beam_groups"].cast<size_t>();
    if (kwargs.contains("num_beams")) config.num_beams = kwargs["num_beams"].cast<size_t>();
    if (kwargs.contains("diversity_penalty")) config.diversity_penalty = kwargs["diversity_penalty"].cast<float>();
    if (kwargs.contains("length_penalty")) config.length_penalty = kwargs["length_penalty"].cast<float>();
    if (kwargs.contains("num_return_sequences")) config.num_return_sequences = kwargs["num_return_sequences"].cast<size_t>();
    if (kwargs.contains("no_repeat_ngram_size")) config.no_repeat_ngram_size = kwargs["no_repeat_ngram_size"].cast<size_t>();
    if (kwargs.contains("stop_criteria")) config.stop_criteria = kwargs["stop_criteria"].cast<StopCriteria>();
    if (kwargs.contains("temperature")) config.temperature = kwargs["temperature"].cast<float>();
    if (kwargs.contains("top_p")) config.top_p = kwargs["top_p"].cast<float>();
    if (kwargs.contains("top_k")) config.top_k = kwargs["top_k"].cast<size_t>();
    if (kwargs.contains("do_sample")) config.do_sample = kwargs["do_sample"].cast<bool>();
    if (kwargs.contains("repetition_penalty")) config.repetition_penalty = kwargs["repetition_penalty"].cast<float>();
    if (kwargs.contains("eos_token_id")) config.eos_token_id = kwargs["eos_token_id"].cast<int64_t>();
}

py::object call_with_config(LLMPipeline& pipe, const std::string& text, const GenerationConfig& config, const StreamerVariant& streamer) {
    if (config.num_return_sequences > 1) {
        py::list res;
        for (auto s: pipe.generate({text}, config, streamer).texts) {
            PyObject* py_s = PyUnicode_DecodeUTF8(s.data(), s.length(), "replace");
            res.append(py_s);
        }
        return res;
    } else {
        auto res = std::string(pipe.generate(text, config, streamer));
        PyObject* py_str = PyUnicode_DecodeUTF8(res.data(), res.length(), "replace");
        return py::reinterpret_steal<py::object>(py_str);
    }
}

std::vector<std::string> call_with_config(LLMPipeline& pipe, const std::vector<std::string>& text, const GenerationConfig& config, const StreamerVariant& streamer) {
    return pipe.generate(text, config, streamer);
}

std::vector<std::string> call_with_kwargs(LLMPipeline& pipeline, const std::vector<std::string>& texts, const py::kwargs& kwargs) {
    GenerationConfig config = pipeline.get_generation_config();
    update_config_from_kwargs(config, kwargs);
    return call_with_config(pipeline, texts, config, kwargs.contains("streamer") ? kwargs["streamer"].cast<StreamerVariant>() : std::monostate());
}

py::object call_with_kwargs(LLMPipeline& pipeline, const std::string& text, const py::kwargs& kwargs) {
    // Create a new GenerationConfig instance and initialize from kwargs
    GenerationConfig config = pipeline.get_generation_config();
    update_config_from_kwargs(config, kwargs);
    return call_with_config(pipeline, text, config, kwargs.contains("streamer") ? kwargs["streamer"].cast<StreamerVariant>() : std::monostate());
}

std::string ov_tokenizers_module_path() {
    // Try a path relative to build artifacts folder first.
    std::filesystem::path from_relative = ov::genai::tokenizers_relative_to_genai();
    if (std::filesystem::exists(from_relative)) {
        return from_relative.string();
    }
    return py::str(py::module_::import("openvino_tokenizers").attr("_ext_path"));
}

class EmptyStreamer: public StreamerBase {
    // It's impossible to create an instance of pure virtual class. Define EmptyStreamer instead.
    bool put(int64_t token) override {
        PYBIND11_OVERRIDE_PURE(
            bool,  // Return type
            StreamerBase,  // Parent class
            put,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    void end() override {
        PYBIND11_OVERRIDE_PURE(void, StreamerBase, end);
    }
};

ov::InferRequest& get_request_from_pyobj(py::object obj) {
    py::str obj_type = py::str(obj.get_type());
    // todo: InferRequest is not accessible from the outside.
    // obj_type is openvino._pyopenvino.InferRequest,
    // which is a pybind binding to InferRequestWrapper (InferRequest is in a m_request field of the latest)
    // and the definition of InferRequestWrapper is not accessible from the outside.

    if (py::isinstance<ov::InferRequest>(obj)) {
        // Directly return the casted object without copying
        return obj.cast<ov::InferRequest&>();
    } else {
        throw std::invalid_argument("Provided object is not castable to ov::InferRequest");
    }
}

} // namespace


PYBIND11_MODULE(py_generate_pipeline, m) {
    m.doc() = "Pybind11 binding for LLM Pipeline";

    py::class_<LLMPipeline>(m, "LLMPipeline")
        .def(py::init([](const std::string& model_path, const std::string& device) {
                ov::genai::ScopedVar env_manager(ov_tokenizers_module_path());
                return std::make_unique<LLMPipeline>(model_path, device);
            }),
        py::arg("model_path"), "path to the model path", 
        py::arg("device") = "CPU", "device on which inference will be done",
        R"(
            LLMPipeline class constructor.
            model_path (str): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
        )")

        .def(py::init<const std::string, const Tokenizer&, const std::string>(), 
        py::arg("model_path"),
        py::arg("tokenizer"),
        py::arg("device") = "CPU",
        R"(
            LLMPipeline class constructor for manualy created openvino_genai.Tokenizer.
            model_path (str): Path to the model file.
            tokenizer (openvino_genai.Tokenizer): tokenizer object.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
        )")

        .def(py::init([](py::object infer_request, 
                            const Tokenizer& tokenizer,
                            OptionalGenerationConfig config) {
            ov::genai::ScopedVar env_manager(ov_tokenizers_module_path());
            return std::make_unique<LLMPipeline>(get_request_from_pyobj(infer_request), tokenizer, config);
        }),
        py::arg("infer_request"), "infer_request", 
        py::arg("tokenizer"), "openvino_genai.Tokenizer object",
        py::arg("config"), "device on which inference will be done")
        .def("generate", py::overload_cast<LLMPipeline&, const std::string&, const py::kwargs&>(&call_with_kwargs),
        R"(
            max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                        `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
            ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
            eos_token_id:  token_id of <eos> (end of sentence)
            
            Beam search specific parameters:
            num_beams:         number of beams for beam search. 1 disables beam search.
            num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
            length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
                `length_penalty` < 0.0 encourages shorter sequences.
            num_return_sequences: the number of sequences to return for grouped beam search decoding.
            no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
            stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values: 
                "EARLY", where the generation stops as soon as there are `num_beams` complete candidates; "HEURISTIC", where an 
                "HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
                "NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
            
            Random sampling parameters:
            temperature:        the value used to modulate token probabilities for random sampling.
            top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
            do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
            repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
        )")
        .def("generate", py::overload_cast<LLMPipeline&, const std::vector<std::string>&, 
                                           const py::kwargs&>(&call_with_kwargs))
        .def("generate", py::overload_cast<LLMPipeline&, const std::vector<std::string>&, 
                                           const GenerationConfig&, const StreamerVariant&>(&call_with_config))
        .def("generate", py::overload_cast<LLMPipeline&, const std::string&, 
                                           const GenerationConfig&, const StreamerVariant&>(&call_with_config))

        .def("__call__", py::overload_cast<LLMPipeline&, const std::string&, 
                                           const py::kwargs&>(&call_with_kwargs))
        .def("__call__", py::overload_cast<LLMPipeline&, const std::string&, 
                                           const GenerationConfig&, const StreamerVariant&>(&call_with_config))
        
        // todo: if input_ids is a ov::Tensor/numpy tensor

        .def("get_tokenizer", &LLMPipeline::get_tokenizer)
        .def("start_chat", &LLMPipeline::start_chat)
        .def("finish_chat", &LLMPipeline::finish_chat)
        .def("get_generation_config", &LLMPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &LLMPipeline::set_generation_config)
        .def("apply_chat_template", &LLMPipeline::apply_chat_template);

     // Binding for Tokenizer
    py::class_<Tokenizer>(m, "Tokenizer",
        R"(openvino_genai.Tokenizer object is used to initialize Tokenizer 
           if it's located in a different path than the main model.)")
        .def(py::init([](const std::string& tokenizer_path) {
            ov::genai::ScopedVar env_manager(ov_tokenizers_module_path());
            return std::make_unique<Tokenizer>(tokenizer_path);
        }), py::arg("tokenizer_path"))
        .def("get_pad_token_id", &Tokenizer::get_pad_token_id)
        .def("get_bos_token_id", &Tokenizer::get_bos_token_id)
        .def("get_eos_token_id", &Tokenizer::get_eos_token_id)
        .def("get_pad_token", &Tokenizer::get_pad_token)
        .def("get_bos_token", &Tokenizer::get_bos_token)
        .def("get_eos_token", &Tokenizer::get_eos_token);

    // Binding for StopCriteria
    py::enum_<StopCriteria>(m, "StopCriteria",
        R"(StopCriteria controls the stopping condition for grouped beam search. The following values are possible:
            "EARLY" stops as soon as there are `num_beams` complete candidates.
            "HEURISTIC" stops when is it unlikely to find better candidates.
            "NEVER" stops when there cannot be better candidates.)")
        .value("EARLY", StopCriteria::EARLY)
        .value("HEURISTIC", StopCriteria::HEURISTIC)
        .value("NEVER", StopCriteria::NEVER)
        .export_values();

     // Binding for GenerationConfig
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &GenerationConfig::ignore_eos)
        .def_readwrite("num_beam_groups", &GenerationConfig::num_beam_groups)
        .def_readwrite("num_beams", &GenerationConfig::num_beams)
        .def_readwrite("diversity_penalty", &GenerationConfig::diversity_penalty)
        .def_readwrite("length_penalty", &GenerationConfig::length_penalty)
        .def_readwrite("num_return_sequences", &GenerationConfig::num_return_sequences)
        .def_readwrite("no_repeat_ngram_size", &GenerationConfig::no_repeat_ngram_size)
        .def_readwrite("stop_criteria", &GenerationConfig::stop_criteria)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("eos_token_id", &GenerationConfig::eos_token_id);

    py::class_<DecodedResults>(m, "DecodedResults")
        .def(py::init<>())
        .def_readwrite("texts", &DecodedResults::texts)
        .def_readwrite("scores", &DecodedResults::scores);

    py::class_<EncodedResults>(m, "EncodedResults")
        .def(py::init<>())
        .def_readwrite("tokens", &EncodedResults::tokens)
        .def_readwrite("scores", &EncodedResults::scores);

    py::class_<StreamerBase, EmptyStreamer, std::shared_ptr<StreamerBase>>(m, "StreamerBase")  // Change the holder form unique_ptr to shared_ptr
        .def(py::init<>())
        .def("put", &StreamerBase::put)
        .def("end", &StreamerBase::end);
}
