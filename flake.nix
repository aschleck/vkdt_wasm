{
  description = "vkdt_wasm development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            bazelisk
            buildifier
            jdk25_headless
            neovim
            nodejs-slim_24
            pnpm
            python3
          ];

          shellHook = ''
            alias bazel=bazelisk
            alias vim=nvim
            export HISTFILESIZE=
            export HISTSIZE=
            export PS1="\[\033[1;32m\][vkdt_wasm:\w]\$\[\033[0m\] "
          '';
        };
      });
}
