{
  description = "Basic C++ project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/23.05";

    utils.url = "github:numtide/flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, ... }@inputs: inputs.utils.lib.eachSystem [
    "x86_64-linux" "i686-linux" "aarch64-linux" "x86_64-darwin"
  ] (system: let pkgs = import nixpkgs {
                   inherit system;
                 };
             in {
               devShell = pkgs.mkShell rec {
                 name = "advcpp ex11";

                 packages = with pkgs; [
                   # Development Tools
                   llvmPackages_16.clang
                   cmake
                   cmakeCurses
                 ];
               };
             });
}
