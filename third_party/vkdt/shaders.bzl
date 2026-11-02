"""Transpiles vkdt's Vulkan GLSL compute shaders to WGSL."""

def vkdt_wgsl(name, module, shaders, srcdir = None, srcs = []):
    """Creates one genrule per shader in src/pipe/modules/<module> of @vkdt, or in srcdir.

    Outputs land at <name>/<module>/<shader>.wgsl, collected in a filegroup called name.

    Args:
      name: the filegroup of all outputs
      module: the vkdt module directory
      shaders: shader basenames without the .comp suffix
      srcdir: the directory of a module of our own in the calling package, instead of vkdt's
      srcs: files the shaders of a module of our own include
    """
    outs = []
    for shader in shaders:
        comp = "%s/%s.comp" % (srcdir, shader) if srcdir else "@vkdt//:src/pipe/modules/%s/%s.comp" % (module, shader)
        out = "%s/%s/%s.wgsl" % (name, module, shader)
        native.genrule(
            name = "%s_%s_%s" % (name, module, shader),
            # shared/ already comes with :common
            srcs = [comp, "@vkdt//:common", "@vkdt//:src/pipe/modules/shared.glsl", "//third_party/vkdt:glsl_fix.pl", "//third_party/vkdt:spv_bindings.pl"] +
                   srcs + ([] if module == "shared" or srcdir else ["@vkdt//:" + module]),
            outs = [out],
            tools = [
                "@glslang//:glslangValidator",
                "@naga_cli//:naga",
                "@spirv_tools//:spirv-opt",
            ],
            # vulkan1.1 (SPIR-V 1.3) is the newest naga translates correctly, and the oldest with
            # subgroup operations.
            cmd = """
                set -e
                comp=$(execpath {comp})
                moddir=$$(dirname $$comp)
                t=$(@D)/{shader}.tmp
                mkdir -p $$t
                # vkdt's modules directory, for shared.glsl and friends
                common=$$(dirname $(execpath @vkdt//:src/pipe/modules/shared.glsl))
                $(execpath @glslang//:glslangValidator) -E -I$$common -I$$(dirname $$moddir) \
                    -I$$moddir $$comp > $$t/pre.comp
                perl $(execpath //third_party/vkdt:glsl_fix.pl) < $$t/pre.comp > $$t/fixed.comp
                $(execpath @glslang//:glslangValidator) -V -S comp --target-env vulkan1.1 \
                    $$t/fixed.comp -o $$t/a.spv > $$t/glslang.log 2>&1 \
                    || {{ cat $$t/glslang.log; exit 1; }}
                $(execpath @spirv_tools//:spirv-opt) --split-combined-image-sampler \
                    $$t/a.spv -o $$t/split.spv
                perl $(location //third_party/vkdt:spv_bindings.pl) $$t/split.spv $$t/b.spv
                $(execpath @naga_cli//:naga) --capabilities all $$t/b.spv $$t/out.wgsl
                # the WebGPU runtime labels pipelines with this, see library_wvk.js
                # naga prints FLT_MAX in full, which WGSL compilers reject as out of f32 range
                # and doesn't enable the subgroups extension it uses. WGSL can't tell that vkdt's
                # subgroup operations run in uniform control flow, which the GLSL requires.
                (echo '// vkdt {module}/{shader}'; \
                 if grep -q subgroup $$t/out.wgsl; then \
                   echo 'enable subgroups;'; echo 'diagnostic(off, subgroup_uniformity);'; fi; \
                 sed -e 's/340282350000000000000000000000000000000f/0x1.fffffep+127f/g' $$t/out.wgsl) > $@
                rm -rf $$t
            """.format(comp = comp, module = module, shader = shader),
        )
        outs.append(out)
    native.filegroup(
        name = name,
        srcs = outs,
    )
