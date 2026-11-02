#!/usr/bin/perl
# Moves each sampler that spirv-opt --split-combined-image-sampler split off a combined image
# sampler to the binding right after its image. glsl_fix.pl doubles every binding number, so that
# slot is always free. spirv-opt's --resolve-binding-conflicts doesn't reliably do this.
#
# usage: spv_bindings.pl <in.spv> <out.spv>

use strict;
use warnings;

my ($in, $out) = @ARGV;
open(my $f, '<:raw', $in) or die "can't read $in: $!";
my $bytes = do { local $/; <$f> };
close($f);
my @w = unpack('V*', $bytes);

# OpName is 5, OpDecorate is 71, and Binding is decoration 33.
my (%name, %binding_at);
for(my $i = 5; $i < @w;) {
  my $count = $w[$i] >> 16;
  my $op = $w[$i] & 0xffff;
  die "malformed SPIR-V" if $count == 0;
  if($op == 5) {
    my $s = pack('V*', @w[$i + 2 .. $i + $count - 1]);
    $s =~ s/\0.*//s;
    $name{$w[$i + 1]} = $s;
  } elsif($op == 71 && $w[$i + 2] == 33) {
    $binding_at{$w[$i + 1]} = $i + 3;
  }
  $i += $count;
}
# Only variables have bindings. The split also gives function parameters these names.
my %by_name = map { $name{$_} => $_ } grep { $binding_at{$_} } keys %name;
for my $id (grep { $binding_at{$_} } keys %name) {
  next unless $name{$id} =~ /^(.*)_sampler$/;
  my $image = $by_name{"$1_image"};
  next unless defined $image && $binding_at{$id} && $binding_at{$image};
  $w[$binding_at{$id}] = $w[$binding_at{$image}] + 1;
}

open(my $o, '>:raw', $out) or die "can't write $out: $!";
print $o pack('V*', @w);
close($o);
