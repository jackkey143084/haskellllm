#!/bin/bash
# Rebuilds the GHC 9.6.6 toolchain in /tmp from the .deb files in /workspace.
# The sandbox intermittently reaps bulk file writes outside persistent volumes,
# so this script is idempotent and fast (local debs, no network). Re-run any
# time a build fails with weird "cannot find" errors.
set -e
S=/tmp/sysroot2
mkdir -p "$S"
cd "$S"
for d in /workspace/*.deb; do dpkg-deb --fsys-tarfile "$d" | tar -x -C .; done
# GHC's Debian layout: real package db + settings live in /var/lib/ghc,
# and /usr/lib/ghc/lib/package.conf.d is a symlink to it.
ln -sfn "$S/var/lib/ghc" /var/lib/ghc
ln -sfn "$S/usr/lib/ghc" /usr/lib/ghc
# linker needs dev symlinks the runtime image lacks
ln -sf libgmp.so.10 /usr/lib/x86_64-linux-gnu/libgmp.so
[ -e "$S/usr/lib/x86_64-linux-gnu/libffi.so" ] || true
ln -sf libffi.so.8 /usr/lib/x86_64-linux-gnu/libffi.so
ln -sf libnuma.so.1 /usr/lib/x86_64-linux-gnu/libnuma.so
# recache package db if cache missing
if [ ! -f "$S/var/lib/ghc/package.conf.d/package.cache" ]; then
  PATH="$S/usr/bin:$PATH" ghc-pkg --global-package-db="$S/var/lib/ghc/package.conf.d" recache
fi
PATH="$S/usr/bin:$PATH" ghc --version
PATH="$S/usr/bin:$PATH" cabal --version
echo "toolchain ready: PATH=$S/usr/bin:\$PATH"
