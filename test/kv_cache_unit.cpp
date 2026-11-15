#include "kv_cache.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    wevision::TextKVCache cache;
    std::string error;
    if (!cache.initialize(2, 4, 2, 3, &error) || cache.length() != 0 ||
        cache.layer_elements() != 24)
        return 1;
    std::vector<float> k0(12), v0(12), k1(12), v1(12);
    for (size_t i = 0; i < k0.size(); ++i) {
        k0[i] = static_cast<float>(i);
        v0[i] = static_cast<float>(100 + i);
        k1[i] = static_cast<float>(200 + i);
        v1[i] = static_cast<float>(300 + i);
    }
    if (!cache.begin_append(2, &error) ||
        !cache.write_layer(0, k0.data(), v0.data(), k0.size(), &error) ||
        cache.commit_append(&error) ||
        !cache.write_layer(1, k1.data(), v1.data(), k1.size(), &error) ||
        !cache.commit_append(&error) || cache.length() != 2 ||
        cache.keys(0)[11] != 11.0F || cache.values(1)[11] != 311.0F)
        return 1;
    if (cache.begin_append(3, &error))
        return 1;
    cache.reset();
    return cache.length() == 0 ? 0 : 1;
}
