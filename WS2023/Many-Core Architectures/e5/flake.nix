{
  description = "CUDA + Python Flake for MCA";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    llvm = pkgs.llvmPackages_latest;
  in {
    devShells.${system}.default = pkgs.mkShell {

      packages = [

        pkgs.gcc
        #builder
        pkgs.cmake
        pkgs.gnumake
        #headers
        pkgs.clang-tools
        #lsp
        llvm.libstdcxxClang
        #tools
        pkgs.cppcheck
        pkgs.valgrind
        pkgs.doxygen

        (pkgs.python3.withPackages (python-pkgs: [
          python-pkgs.numpy
          python-pkgs.pandas
          python-pkgs.scipy
          python-pkgs.matplotlib
          python-pkgs.requests
          python-pkgs.debugpy
          python-pkgs.python-lsp-server
        ]))
      ];
    };
  };
}
