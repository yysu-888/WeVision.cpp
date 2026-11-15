#ifndef WEVISION_MASK_HPP
#define WEVISION_MASK_HPP

#include "runtime.hpp"

#include <vector>

namespace wevision {

// Structured tags carried by graph profile points. Consumers dispatch on the
// tag instead of parsing point names. Values are opaque to GraphSession.
enum MaskProfileTag {
    MaskProfileNone = 0,
    MaskProfileEncoder = 1,
    MaskProfileFpn = 2,
    MaskProfilePixelFusion = 3,
    MaskProfileTransformerWindow = 4, // msda sampler interval start
    MaskProfileMsdaEnd = 5,           // msda sampler interval end
    MaskProfileTransformer = 6,
    MaskProfileCoarse = 7,
    MaskProfileAlphaS4 = 8,
    MaskProfileAlphaS2 = 9,
    MaskProfileAlphaS1 = 10,
    MaskProfileAlphaS1Detail = 11, // s1 refinement interval inside logits_s1
};

struct MaskQueryOutput {
    ggml_tensor *content = nullptr;
    ggml_tensor *position = nullptr;
};

struct MaskProjectedFeatures {
    std::vector<ggml_tensor *> maps;
};

struct MaskEncoderInputs {
    ggml_tensor *src = nullptr; // [hidden, all_level_tokens]
    ggml_tensor *pos = nullptr; // [hidden, all_level_tokens]
    std::vector<int32_t> heights;
    std::vector<int32_t> widths;
};

struct MaskFpnOutput {
    ggml_tensor *feature = nullptr;
}; // [W,H,hidden] WHC working layout (see build_mask_fpn)

struct MaskAlphaOutput {
    ggml_tensor *logits_s1 = nullptr;
    ggml_tensor *logits_s2 = nullptr;
    // Reachable pre-reshape boundaries used for backend stage profiling.
    ggml_tensor *stage_s4 = nullptr;
    ggml_tensor *stage_s2 = nullptr;
};

// Object-independent projections shared across all objects in a multi-object
// mask pass.
struct MaskAlphaShared {
    ggml_tensor *semantic_s4 = nullptr;
    ggml_tensor *semantic_s2 = nullptr;
    ggml_tensor *semantic_s1 = nullptr;
    ggml_tensor *rgb_s4 = nullptr;
    ggml_tensor *rgb_s2 = nullptr;
    ggml_tensor *rgb_s1 = nullptr;
};

// boxes are normalized xyxy, object-major [object_count, 4]. obj_hidden is
// [text.hidden_size, object_count].
bool build_mask_query(GraphSession &session, ggml_tensor *obj_hidden,
                      const std::vector<float> &boxes, MaskQueryOutput *output,
                      std::string *error);

bool build_mask_encoder_projection(GraphSession &session,
                                   const std::vector<ggml_tensor *> &features,
                                   MaskProjectedFeatures *output,
                                   std::string *error);

// Flattens projected C,W,H maps in Python's H,W token order, adds the learned
// level embedding, and creates DINO's normalized 2-D sine/cosine position.
bool build_mask_encoder_inputs(GraphSession &session,
                               const MaskProjectedFeatures &projected,
                               MaskEncoderInputs *output, std::string *error);

// Executes the released SpatialConvEncoder stack over the flattened DINO
// inputs. `layer_count` is intentionally explicit while the GGUF v1 metadata
// does not yet carry the encoder-layer count.
bool build_mask_spatial_encoder(GraphSession &session,
                                const MaskEncoderInputs &input,
                                int32_t layer_count, ggml_tensor **output,
                                std::string *error);

// Runs the six-layer DINO mask decoder. Query tensors are [hidden, objects],
// memory is [hidden, tokens], and boxes are normalized xyxy object-major.
bool build_mask_transformer(
    GraphSession &session, ggml_tensor *content, ggml_tensor *position,
    const std::vector<float> &boxes, ggml_tensor *memory,
    const MaskEncoderInputs &spatial, ggml_tensor **output, std::string *error,
    std::vector<GraphProfilePoint> *profile_points = nullptr);

bool build_mask_fpn(GraphSession &session, ggml_tensor *memory,
                    const MaskEncoderInputs &spatial, MaskFpnOutput *output,
                    std::string *error);

// Runs AdditivePixelFusion from an already decoded FPN feature and the
// decoder RGB input. The fused pixel feature is returned in contiguous W,H,C
// layout so the WHC alpha cascade can consume it without a layout copy.
// build_dynamic_mask_predictor converts it to C,W,H once at its entry for the
// per-pixel GEMM contraction.
bool build_mask_pixel_fusion(GraphSession &session, ggml_tensor *fpn_feature,
                             ggml_tensor *rgb, ggml_tensor **output,
                             std::string *error);

bool build_mask_alpha_shared(GraphSession &session, ggml_tensor *rgb,
                             ggml_tensor *pixel_feature,
                             MaskAlphaShared *output, std::string *error);

// Executes FullCanvasAlphaCascadeHead. `coarse_logits` is W,H,objects;
// `pixel_feature` must be W,H,C (see build_mask_pixel_fusion); returned logits
// preserve that layout at S/1 and S/2 respectively.
bool build_mask_alpha_cascade(
    GraphSession &session, ggml_tensor *rgb, ggml_tensor *pixel_feature,
    ggml_tensor *queries, ggml_tensor *coarse_logits, MaskAlphaOutput *output,
    std::string *error,
    std::vector<GraphProfilePoint> *profile_points = nullptr,
    const MaskAlphaShared *shared_alpha = nullptr);

// `pixel_feature` must be W,H,C (see build_mask_pixel_fusion).
bool build_dynamic_mask_predictor(GraphSession &session, ggml_tensor *queries,
                                  ggml_tensor *pixel_feature,
                                  ggml_tensor **output, std::string *error);

// Internal deterministic fixture boundary for validating the phase-convolution
// decomposition against ggml's ConvTranspose2d implementation.
bool build_mask_conv_transpose2x_fixture(
    GraphSession &session, ggml_tensor *input, const std::string &weight_name,
    bool phase_decomposition, ggml_tensor **output, std::string *error);

// Internal fixture boundary exposing the N==1 fast path of conv2d_whc_weight
// for unit-test validation against ggml_conv_2d.
bool build_mask_conv2d_whc_fixture(GraphSession &session, ggml_tensor *input,
                                   ggml_tensor *weight, int padding, int stride,
                                   ggml_tensor **output, std::string *error);

} // namespace wevision
#endif
