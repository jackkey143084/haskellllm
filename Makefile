# Tiny-LLM-in-Haskell build. Toolchain comes from tools/setup-toolchain.sh
# (GHC 9.6.6 extracted from the local .deb set into /tmp/sysroot2).

GHC := /tmp/sysroot2/usr/bin/ghc
GHCOPTS := -O2 -Wall -isrc -itest -L/tmp/sysroot2/usr/lib/x86_64-linux-gnu

.PHONY: test inspect clean setup

setup:
	bash tools/setup-toolchain.sh

test: setup
	$(GHC) $(GHCOPTS) -O0 -o /tmp/hq-selftest test/selftest.hs && /tmp/hq-selftest

# Inspect a real safetensors shard: make inspect SHARD=/opt/models/.../model.safetensors
inspect: setup
	$(GHC) $(GHCOPTS) -O0 -o /tmp/hq-inspect app/Inspect.hs && /tmp/hq-inspect $(SHARD)

clean:
	rm -f /tmp/hq-selftest /tmp/hq-inspect
