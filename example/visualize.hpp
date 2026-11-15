#ifndef WEVISION_EXAMPLE_VISUALIZE_HPP
#define WEVISION_EXAMPLE_VISUALIZE_HPP

#include "wevision.h"

#include <string>

// Draws the prediction overlay (mask/alpha tint, boxes, points, labels) onto
// an RGB copy of the source image and writes it as a PNG to `path`.
//
// `fallback_label` replaces empty/null labels in the annotation text.
// Returns false when the path is empty, the buffer is invalid, or the PNG
// cannot be written; a successful write also prints "visualization=<path>".
bool write_visualization(const std::string &path, const wevision_result &result,
                         const std::string &fallback_label,
                         const unsigned char *rgb, int width, int height);

#endif
