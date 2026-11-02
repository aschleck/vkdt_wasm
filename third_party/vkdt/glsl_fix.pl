#!/usr/bin/perl
# Rewrites preprocessed vkdt GLSL into the subset that naga can turn into WGSL.
use strict; use warnings;
local $/; my $src = <STDIN>;
# WGSL storage textures need a format, Vulkan's don't. The WebGPU runtime swaps in the real one.
$src =~ s{layout\s*\(([^)]*)\)(\s*uniform\s+(?:(?:writeonly|readonly|coherent|volatile|restrict)\s+)*u?image2D(?:Array)?\b)}{
  my ($l, $rest) = ($1, $2);
  ($l =~ /\b(?:rgba|rg|r)(?:8|16|32)(?:f|i|ui|_snorm)?\b|r11f_g11f_b10f/) ? "layout($l)$rest" : "layout($l, rgba16f)$rest"
}ge;
# Splitting combined image samplers needs a free binding for each sampler. Doubling every binding
# number gives each one a predictable home: Vulkan binding B becomes WGSL binding 2B, and a combined
# image sampler's sampler lands on 2B + 1.
$src =~ s{(layout\s*\([^)]*\bbinding\s*=\s*)(\d+)}{$1 . (2 * $2)}ge;
# WGSL has no push constants. A std430 storage buffer keeps their memory layout, which a uniform
# block wouldn't, and the WebGPU runtime binds the pushed bytes there for each dispatch.
$src =~ s{layout\s*\(\s*push_constant\s*(?:,[^)]*)?\)\s*uniform}{layout(std430, set = 3, binding = 0) readonly buffer}g;
# WebGPU has no arrays of bindings, but vkdt's arrays hold images of one size (llap's gamma
# levels, for instance), so the WebGPU runtime binds them as layers of one array texture. Here we
# turn image arrays into array images and every access to element i into an access to layer i.
my %arrays;
$src =~ s{\b(u?sampler2D|u?image2D)(\s+)(\w+)\s*\[\s*\d*\s*\]}{$arrays{$3} = 1; "$1Array$2$3"}ge;
$src = layer_arrays($src, \%arrays) if %arrays;
# WGSL has no image atomics, so images that see them become storage buffers of u32 with their
# size up front, and the WebGPU runtime syncs those with the images around each dispatch.
my %atomic = map { $_ => 1 } ($src =~ /\bimageAtomic\w*\s*\(\s*(\w+)/g);
for my $name (keys %atomic) {
  next unless $src =~ s{layout\s*\(([^)]*)\)\s*uniform\s+(?:(?:coherent|volatile|restrict)\s+)*uimage2D\s+\Q$name\E\s*;}{
    my $l = $1;
    my ($set) = $l =~ /\bset\s*=\s*(\d+)/;
    my ($binding) = $l =~ /\bbinding\s*=\s*(\d+)/;
    $set //= 0;
    my $idx = "uint(p.y) * $name.wasm_wd + uint(p.x)";
    my $out = "any(greaterThanEqual(uvec2(p), uvec2($name.wasm_wd, $name.wasm_ht)))";
    my $h = "layout(std430, set = $set, binding = $binding) buffer wasm_atomic_${name}_t { uint wasm_wd; uint wasm_ht; uint data[]; } $name;\n";
    for my $op (qw(Add Min Max And Or Xor Exchange)) {
      $h .= "uint wasm_imageAtomic${op}_$name(ivec2 p, uint v) { if($out) return 0u; return atomic$op($name.data[$idx], v); }\n";
    }
    $h .= "uint wasm_imageAtomicCompSwap_$name(ivec2 p, uint c, uint v) { if($out) return 0u; return atomicCompSwap($name.data[$idx], c, v); }\n";
    $h .= "uvec4 wasm_imageLoad_$name(ivec2 p) { if($out) return uvec4(0); return uvec4($name.data[$idx], 0u, 0u, 1u); }\n";
    $h .= "void wasm_imageStore_$name(ivec2 p, uvec4 v) { if($out) return; $name.data[$idx] = v.x; }\n";
    $h;
  }e;
  $src = map_calls($src, sub {
    my ($f, @args) = @_;
    return unless @args && $args[0] =~ /^\s*\Q$name\E\s*$/;
    return "ivec2($name.wasm_wd, $name.wasm_ht)" if $f eq 'imageSize';
    return unless $f =~ /^(imageAtomic\w+|imageLoad|imageStore)$/;
    return "wasm_${f}_$name(" . join(',', @args[1..$#args]) . ")";
  });
}
# WGSL's subgroups extension has no subgroup ids, counts, or barriers. GPUs lay subgroups out along
# the local invocation index, so the ids follow from it, and a workgroup barrier is a stronger
# subgroup barrier where control flow allows one.
$src =~ s/\bgl_SubgroupID\b/(gl_LocalInvocationIndex \/ gl_SubgroupSize)/g;
# naga's SPIR-V frontend has no elect, which is the lowest active lane, and broadcasting it needs
# the ballot extension
if($src =~ s/\bsubgroupElect\s*\(\s*\)/(subgroupBroadcastFirst(gl_SubgroupInvocationID) == gl_SubgroupInvocationID)/g) {
  $src =~ s/^(#version[^\n]*\n)/$1#extension GL_KHR_shader_subgroup_ballot : enable\n/m;
}
# naga also refuses the invocation id builtin in multi dimensional workgroups.
$src =~ s/\bgl_SubgroupInvocationID\b/(gl_LocalInvocationIndex % gl_SubgroupSize)/g;
$src =~ s/\bgl_NumSubgroups\b/((gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z + gl_SubgroupSize - 1u) \/ gl_SubgroupSize)/g;
$src =~ s/\bsubgroupBarrier\s*\(\s*\)/barrier()/g;
# An if block that ends in a barrier only compiles in WGSL when the condition is provably uniform,
# such as when it reads shared memory after a barrier. Running the barrier after the block instead
# is the same for every invocation when the condition is uniform, which GLSL requires anyway.
$src =~ s{(\bif\s*\((?:[^()]|\([^()]*\))*\)\s*\{(?:[^{}]|\{[^{}]*\})*?)\bbarrier\s*\(\s*\)\s*;(\s*\})}{$1$2 barrier();}g;
# WGSL only allows barriers in uniform control flow, so a bounds check that returns early before a
# barrier() won't compile. Dropping those checks is safe because WebGPU makes out of bounds image
# loads return zero and out of bounds stores do nothing.
if($src =~ /\bbarrier\s*\(\s*\)/) {
  my $paren = qr/(\((?:[^()]++|(?-1))*\))/;
  if($src =~ /\bvoid\s+main\s*\(\s*\)\s*\{/g) {
    my $body = pos($src);
    my $first = index($src, 'barrier()', $body);
    if($first > 0) {
      my $head = substr($src, $body, $first - $body);
      $head =~ s/\bif\s*$paren\s*return\s*;//g;
      substr($src, $body, $first - $body) = $head;
    }
  }
}
# naga mistranslates functions that can run off their end without returning, which GLSL allows
# when the result is undefined, so they get an explicit return of zero.
{
  my $type = qr/(?:float|int|uint|bool|[iub]?vec[234]|mat[234](?:x[234])?)/;
  my $out = '';
  while($src =~ /\b($type)\s+\w+\s*\([^()]*\)\s*\{/g) {
    my ($ret, $open) = ($1, $+[0]);
    my ($depth, $i) = (1, $open);
    for(; $i < length($src) && $depth; $i++) {
      my $c = substr($src, $i, 1);
      $depth++ if $c eq '{';
      $depth-- if $c eq '}';
    }
    my $close = $i - 1;
    my $body = substr($src, $open, $close - $open);
    $out .= substr($src, 0, $close);
    $out .= "  return $ret(0);\n" unless $body =~ /\breturn\b[^;]*;\s*$/;
    $src = substr($src, $close);
    pos($src) = 0;
  }
  $src = $out . $src;
}
# naga's SPIR-V frontend has no OpIsNan or OpIsInf.
my @types = (['float', 'bool'], ['vec2', 'bvec2'], ['vec3', 'bvec3'], ['vec4', 'bvec4']);
my $helpers = '';
for my $t (@types) {
  my ($v, $b) = @$t;
  my $nan = $v eq 'float' ? 'x != x' : 'notEqual(x, x)';
  my $inf = $v eq 'float' ? 'abs(x) > 3.4e38' : "greaterThan(abs(x), $v(3.4e38))";
  $helpers .= "$b wasm_isnan($v x) { return $nan; }\n$b wasm_isinf($v x) { return $inf; }\n";
}
$src =~ s/\bisnan\s*\(/wasm_isnan(/g;
$src =~ s/\bisinf\s*\(/wasm_isinf(/g;
# after the last #extension, since functions can't come before those
if($src =~ /(.*#extension[^\n]*\n)/s) { substr($src, length($1), 0) = $helpers; }
else { $src =~ s/(#version[^\n]*\n)/$1$helpers/; }
print $src;

# Splits the comma separated arguments in $s at depth zero.
sub split_args {
  my ($s) = @_;
  my (@args, $depth, $cur);
  $depth = 0; $cur = '';
  for my $c (split //, $s) {
    if($c eq ',' && $depth == 0) { push @args, $cur; $cur = ''; next; }
    $depth++ if $c eq '(' || $c eq '[';
    $depth-- if $c eq ')' || $c eq ']';
    $cur .= $c;
  }
  push @args, $cur if $cur ne '' || @args;
  return @args;
}

# Calls $f->(name, args) for every call in $s, innermost first, and replaces the call with what
# it returns unless that's undef.
sub map_calls {
  my ($s, $f) = @_;
  my $out = '';
  while($s =~ /\b(\w+)\s*\(/g) {
    my ($name, $start, $open) = ($1, $-[0], $+[0]);
    my ($depth, $i) = (1, $open);
    for(; $i < length($s) && $depth; $i++) {
      my $c = substr($s, $i, 1);
      $depth++ if $c eq '(';
      $depth-- if $c eq ')';
    }
    my $inner = map_calls(substr($s, $open, $i - 1 - $open), $f);
    my @args = split_args($inner);
    my $r = $f->($name, @args);
    $out .= substr($s, 0, $start) . (defined $r ? $r : "$name($inner)");
    $s = substr($s, $i);
  }
  return $out . $s;
}

# Rewrites accesses to elements of the arrays in %$arrays, and to sampler parameters of helper
# functions that get passed such elements, into accesses to layers.
sub layer_arrays {
  my ($src, $arrays) = @_;
  # how to find the layer of an argument, given the names of layered parameters in scope
  my $layer_of = sub {
    my ($arg, $params) = @_;
    return ($1, $2) if $arg =~ /^\s*(\w+)\s*\[(.*)\]\s*$/s && $arrays->{$1};
    return ($1, "wasm_layer_$1") if $arg =~ /^\s*(\w+)\s*$/ && $params->{$1};
    return;
  };
  my %ops = (
    texelFetch   => sub { my ($t, $l, @a) = @_; "texelFetch($t, ivec3($a[0], $l), $a[1])" },
    texture      => sub { my ($t, $l, @a) = @_; "texture($t, vec3($a[0], $l)" . join('', map { ", $_" } @a[1..$#a]) . ")" },
    textureLod   => sub { my ($t, $l, @a) = @_; "textureLod($t, vec3($a[0], $l), $a[1])" },
    textureGather=> sub { my ($t, $l, @a) = @_; "textureGather($t, vec3($a[0], $l)" . join('', map { ", $_" } @a[1..$#a]) . ")" },
    textureSize  => sub { my ($t, $l, @a) = @_; "textureSize($t, $a[0]).xy" },
    imageStore   => sub { my ($t, $l, @a) = @_; "imageStore($t, ivec3($a[0], $l), $a[1])" },
    imageLoad    => sub { my ($t, $l, @a) = @_; "imageLoad($t, ivec3($a[0], $l))" },
    imageSize    => sub { my ($t, $l, @a) = @_; "imageSize($t).xy" },
  );

  # helper functions taking samplers, by name, with their parameter lists and bodies
  my %helpers;
  while($src =~ /\b(\w+)\s+(\w+)\s*\(([^()]*\b(?:u?sampler2D)\s+\w+[^()]*)\)\s*\{/g) {
    my ($ret, $name, $params, $body_start) = ($1, $2, $3, $+[0]);
    next if $ret eq 'return' || $ret eq 'else';
    my ($depth, $i) = (1, $body_start);
    for(; $i < length($src) && $depth; $i++) {
      my $c = substr($src, $i, 1);
      $depth++ if $c eq '{';
      $depth-- if $c eq '}';
    }
    $helpers{$name} = {ret => $ret, params => $params, body => substr($src, $body_start, $i - 1 - $body_start), end => $i};
  }

  my %needed;  # helper name => 1 when called with a layered argument
  my $rewrite;
  $rewrite = sub {
    my ($code, $params) = @_;
    return map_calls($code, sub {
      my ($name, @args) = @_;
      return unless @args;
      if($ops{$name}) {
        my ($t, $l) = $layer_of->($args[0], $params);
        return unless defined $t;
        return $ops{$name}->($t, $l, @args[1..$#args]);
      }
      if($helpers{$name}) {
        my $changed = 0;
        my @out = map {
          my ($t, $l) = $layer_of->($_, $params);
          # layers are often uint, which GLSL won't convert to int when resolving overloads
          defined $t ? do { $changed = 1; "$t, int($l)" } : $_
        } @args;
        return unless $changed;
        $needed{$name} = 1;
        return "$name(" . join(', ', @out) . ")";
      }
      return;
    });
  };

  # Rewrite the body of main and everything else outside helper definitions.
  $src = $rewrite->($src, {});

  # Then give each helper that receives layers an overload taking them, until no new ones appear.
  my %done;
  while(my @todo = grep { !$done{$_} } keys %needed) {
    for my $name (@todo) {
      $done{$name} = 1;
      my $h = $helpers{$name};
      my %params;
      (my $params = $h->{params}) =~ s{\b(u?sampler2D)\s+(\w+)}{$params{$2} = 1; "$1Array $2, int wasm_layer_$2"}ge;
      my $body = $rewrite->($h->{body}, \%params);
      my $overload = "\n$h->{ret} $name($params) {$body}\n";
      # right after the original, which every caller already comes after
      if($src =~ /\b\Q$h->{ret}\E\s+\Q$name\E\s*\(\Q$h->{params}\E\)\s*\{/g) {
        my ($depth, $i) = (1, $+[0]);
        for(; $i < length($src) && $depth; $i++) {
          my $c = substr($src, $i, 1);
          $depth++ if $c eq '{';
          $depth-- if $c eq '}';
        }
        substr($src, $i, 0) = $overload;
      }
    }
  }
  return $src;
}
