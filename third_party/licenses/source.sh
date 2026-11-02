#!/bin/sh
# Writes the notice that tells recipients where to get the engine's source, as LGPL 2.1 section 6
# requires for the rawspeed linked into it. Takes MODULE.bazel, so the rawspeed version it names
# is always the one the engine was built with.
set -eu

url=$(grep -o 'https://github.com/darktable-org/rawspeed/archive/[0-9a-f]*\.tar\.gz' "$1")
sha=$(grep -A4 'name = "rawspeed"' "$1" | sed -n 's/.*sha256 = "\([0-9a-f]*\)".*/\1/p')
test -n "$url" && test -n "$sha"

cat <<EOF
==== Source code ====

This engine statically links rawspeed, which is under the GNU LGPL 2.1 or later. You may get the
complete source of the rawspeed it contains, and the source of the rest of the engine so that you
can modify rawspeed and rebuild the engine with it, from:

* rawspeed, unmodified: $url
  (sha256 $sha)
* vkdt_wasm, which builds the engine and configures rawspeed for it in
  third_party/rawspeed/BUILD.rawspeed.bazel: https://github.com/aschleck/vkdt_wasm
  MODULE.bazel there pins the exact version of everything else the engine contains.

To rebuild the engine against a modified rawspeed, point the rawspeed http_archive in
vkdt_wasm's MODULE.bazel at your copy and run \`bazel build //:dist\`.
EOF
