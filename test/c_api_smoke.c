#include "wevision.h"

#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    wevision_model *model = NULL;
    if (wevision_model_load(argv[1], &model) != WEVISION_OK) {
        fprintf(stderr, "%s\n", wevision_last_error());
        return 1;
    }
    wevision_runtime_options options = {WEVISION_BACKEND_CPU, 512, 32};
    wevision_runtime *runtime = NULL;
    const wevision_status status =
        wevision_runtime_create(model, &options, &runtime);
    wevision_model_free(model);
    if (status != WEVISION_OK) {
        fprintf(stderr, "%s\n", wevision_last_error());
        return 1;
    }
    wevision_runtime_free(runtime);
    return 0;
}
