cmake_minimum_required(VERSION 3.14)
project(wevision LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WEVISION_GGML_METAL "Enable the ggml Metal backend" OFF)
option(WEVISION_GGML_CUDA "Enable the ggml CUDA backend" OFF)
option(WEVISION_BUILD_TESTS "Build reproducible WeVision validation tools" ON)

set(GGML_METAL ${WEVISION_GGML_METAL} CACHE BOOL "" FORCE)
set(GGML_CUDA ${WEVISION_GGML_CUDA} CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# AppleClang can accept OpenMP flags without providing the matching runtime
# library. Keep the portable ggml thread pool for this standalone project.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

add_library(wevision STATIC
    src/model_loader.cpp
    src/runtime.cpp
    src/trace.cpp
    src/profile.cpp
    src/vision.cpp
    src/tokenizer.cpp
    src/text.cpp
    src/kv_cache.cpp
    src/mask.cpp
    src/mask_encoder.cpp
    src/mask_ops.cpp
    src/ms_deform_attn.cpp
    src/pixel_ops.cpp
    src/postprocess.cpp
    src/inference.cpp
    src/wevision_capi.cpp
)

target_include_directories(wevision
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(wevision PRIVATE ggml)
if(WEVISION_GGML_METAL)
    target_compile_definitions(wevision PRIVATE WEVISION_USE_METAL)
endif()
if(WEVISION_GGML_CUDA)
    target_compile_definitions(wevision PRIVATE WEVISION_USE_CUDA)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(wevision PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(Threads REQUIRED)

add_executable(wevision-cli
    example/wevision_cli.cpp
    example/visualize.cpp
)
target_include_directories(wevision-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/example
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples
)
target_link_libraries(wevision-cli PRIVATE wevision)

# HTTP inference service and its client example (cpp-httplib, header-only).
add_executable(wevision-serve example/wevision_serve.cpp)
target_include_directories(wevision-serve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/example
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples
)
target_link_libraries(wevision-serve PRIVATE wevision Threads::Threads)

add_executable(wevision-client example/wevision_client.cpp)
target_include_directories(wevision-client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/example
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib
)
target_link_libraries(wevision-client PRIVATE Threads::Threads)

# Offline quantizer tool: only depends on ggml/gguf, no inference code.
add_executable(wevision-quantize tools/quantize.cpp)
target_include_directories(wevision-quantize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
)
target_link_libraries(wevision-quantize PRIVATE ggml)

if(WEVISION_BUILD_TESTS)
    enable_testing()

    function(wevision_add_test_tool target source)
        add_executable(${target} ${source})
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
        )
        target_link_libraries(${target} PRIVATE wevision)
    endfunction()

    wevision_add_test_tool(wevision-inspect test/inspect_model.cpp)
    wevision_add_test_tool(wevision-trace-smoke test/trace_smoke.cpp)
    wevision_add_test_tool(wevision-vision-unit test/vision_unit.cpp)
    wevision_add_test_tool(wevision-vision-dump test/vision_dump.cpp)
    wevision_add_test_tool(wevision-vision-patch-smoke test/vision_patch_smoke.cpp)
    wevision_add_test_tool(wevision-vision-stack-smoke test/vision_stack_smoke.cpp)
    wevision_add_test_tool(wevision-tokenizer-dump test/tokenizer_dump.cpp)
    wevision_add_test_tool(wevision-text-embedding-smoke test/text_embedding_smoke.cpp)
    wevision_add_test_tool(wevision-text-position-dump test/text_position_dump.cpp)
    wevision_add_test_tool(wevision-text-block0-qkv-smoke test/text_block0_qkv_smoke.cpp)
    wevision_add_test_tool(wevision-text-block0-full-smoke test/text_block0_full_smoke.cpp)
    wevision_add_test_tool(wevision-text-prefill-smoke test/text_prefill_smoke.cpp)
    wevision_add_test_tool(wevision-text-decode-smoke test/text_decode_smoke.cpp)
    wevision_add_test_tool(wevision-kv-cache-unit test/kv_cache_unit.cpp)
    wevision_add_test_tool(wevision-mask-query-smoke test/mask_query_smoke.cpp)
    wevision_add_test_tool(wevision-mask-projection-smoke test/mask_projection_smoke.cpp)
    wevision_add_test_tool(wevision-mask-encoder-inputs-smoke test/mask_encoder_inputs_smoke.cpp)
    wevision_add_test_tool(wevision-mask-transformer-smoke test/mask_transformer_smoke.cpp)
    wevision_add_test_tool(wevision-mask-fpn-smoke test/mask_fpn_smoke.cpp)
    wevision_add_test_tool(wevision-mask-predictor-smoke test/mask_predictor_smoke.cpp)
    wevision_add_test_tool(wevision-mask-pixel-fusion-smoke test/mask_pixel_fusion_smoke.cpp)
    wevision_add_test_tool(wevision-mask-alpha-smoke test/mask_alpha_smoke.cpp)
    wevision_add_test_tool(wevision-mask-conv-transpose-unit test/mask_conv_transpose_unit.cpp)
    wevision_add_test_tool(wevision-conv2d-whc-unit test/conv2d_whc_unit.cpp)
    wevision_add_test_tool(wevision-pixel-ops-smoke test/pixel_ops_smoke.cpp)
    wevision_add_test_tool(wevision-msda-sampler-smoke test/msda_sampler_smoke.cpp)
    wevision_add_test_tool(wevision-postprocess-unit test/postprocess_unit.cpp)

    add_executable(wevision-c-api-smoke test/c_api_smoke.c)
    target_link_libraries(wevision-c-api-smoke PRIVATE wevision)
    set_target_properties(wevision-c-api-smoke PROPERTIES LINKER_LANGUAGE CXX)
    wevision_add_test_tool(wevision-predict-smoke test/predict_smoke.cpp)

    add_test(NAME wevision-vision-unit COMMAND wevision-vision-unit)
    add_test(NAME wevision-kv-cache-unit COMMAND wevision-kv-cache-unit)
    add_test(NAME wevision-postprocess-unit COMMAND wevision-postprocess-unit)
endif()
