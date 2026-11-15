#!/usr/bin/env sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
formatter=${CLANG_FORMAT:-clang-format}

if ! command -v "$formatter" >/dev/null 2>&1; then
    echo "error: clang-format was not found; install it or set CLANG_FORMAT to its path" >&2
    exit 1
fi

find "$project_root/src" "$project_root/include" "$project_root/test" "$project_root/tools" \
    -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) \
    -exec "$formatter" -i -style=file {} +

echo "formatted project source files"
