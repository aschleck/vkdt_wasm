"""The vkdt modules we build, shared by the shader and engine builds."""

# name: (the source holding the module's callbacks or None, compute shaders to transpile)
VKDT_MODULES = {
    # vkdt's graph machinery
    "display": (None, []),
    "mipmap": (None, ["main"]),
    "shared": (None, ["blur", "blurh", "blurs", "blurv", "guided1", "guided1f", "guided2", "guided2f", "guided3", "resample", "scatter"]),

    # input and output
    "i-raw": ("main.cc", []),
    "colenc": ("main.c", ["main"]),
    "resize": ("main.c", ["main"]),

    # analysis
    "hist": ("main.c", ["collect", "map"]),
    # collect needs float atomics, which WebGPU lacks, so vkdt picks coldumb instead
    "pick": ("main.c", ["coldumb", "display"]),

    # raw
    "hotpx": ("main.c", ["main"]),
    "denoise": ("main.c", ["assemble", "doub", "down", "downcov", "half", "noop"]),
    "hilite": ("main.c", ["assemble", "doub", "half", "reduce"]),
    "demosaic": ("main.c", ["down", "fix", "gauss", "halfsize", "rcd_conv", "rcd_fill", "splat"]),

    # scene referred
    "lens": (None, ["main"]),
    "crop": ("main.c", ["main"]),
    "ca": ("main.c", ["assemble", "reduce"]),
    "colour": ("main.c", ["atemp-", "autotemp", "main", "main-"]),
    "exposure": (None, ["main"]),
    "negative": ("main.c", ["main"]),
    "dehaze": ("main.c", ["dehaze", "depth"]),
    "grad": (None, ["main"]),
    "vignette": (None, ["main"]),
    "zones": ("main.c", ["apply", "quant"]),
    "eq": ("main.c", ["down", "up"]),
    "contrast": ("main.c", ["combine"]),

    # tone mapping
    # Not dspy, its curve plot, which includes the deleted shared/oetf.glsl. vkdt only loads the
    # shaders of nodes that lead to a sink, and our graphs leave dspy unconnected.
    "filmcurv": ("main.c", ["main"]),
    "reinhard": (None, ["main"]),

    # display referred
    "llap": ("main.c", ["assemble", "colour", "curve", "reduce"]),
    "curves": ("main.c", ["dspy", "hist", "main", "prep"]),
    "mask": ("main.c", ["dspy", "hist", "main"]),
    "blend": ("main.c", ["down", "main", "up"]),
    "hdrmerge": ("main.c", ["linear", "map"]),
    "i-lut": ("main.c", []),
    "filmsim": ("main.c", ["develop", "dirlut", "expose", "fast", "halin", "halout", "negprint", "scatter", "setup"]),
    "align": ("main.c", ["dist", "down2", "down4", "half", "mask", "merge", "warp"]),
    "grade": (None, ["main"]),
    # Not dspy, whose number drawing comes from a Shadertoy shader with no license of its own. vkdt
    # never loads it, as with filmcurv's.
    "okgmtclp": ("main.c", ["main"]),
    "deconv": ("main.c", ["deconv"]),
    "usm": (None, ["main"]),
    "grain": (None, ["main"]),
    "frame": ("main.c", ["main"]),
}

# Modules without params or params.ui files.
VKDT_NO_PARAMS = ["display", "hist"]

# Other module directories whose files a module's shaders include
SHADER_INCLUDES = {
    "filmsim": ["rt"],
}
