#!/usr/bin/env bash
# KNOX-SWA-UNDER-MLA
#
# vllm/model_executor/layers/attention/attention.py, Attention.get_kv_cache_spec:
#
#     if self.sliding_window is not None:
#         assert not vllm_config.model_config.use_mla, \
#             "MLA is not supported for slidingwindow"
#
# `model_config.use_mla` describes the TARGET model. This method lives on the
# non-MLA `Attention` class (MLA layers use MLAAttention in mla_attention.py),
# so a sliding-window layer reaching here is genuinely non-MLA. The DFlash2
# drafter registers 6 plain sliding-window layers under an MLA target, which the
# assert predates and wrongly rejects. Dropping it lets those layers get the
# SlidingWindowSpec they actually need; MLA target layers never reach this line.
set -uo pipefail
IMG=vllm-glm52-b12x:dflash2-port
REL=model_executor/layers/attention/attention.py
OUT="$HOME/patches-port/attention.py"
LAUNCH="$HOME/launch-glm53-dflash2port.sh"
MARK=KNOX-SWA-UNDER-MLA
mkdir -p "$HOME/patches-port"

if [ ! -f "$OUT" ] || ! grep -q "$MARK" "$OUT" 2>/dev/null; then
  CID=$(docker create "$IMG" 2>/dev/null)
  docker cp "$CID:/usr/local/lib/python3.12/dist-packages/vllm/$REL" "$OUT" >/dev/null 2>&1
  docker rm -f "$CID" >/dev/null 2>&1
  [ -f "$OUT" ] || { echo "EXTRACT FAILED"; exit 1; }
  python3 - "$OUT" "$MARK" << 'PYEOF'
import ast, sys
p, mark = sys.argv[1], sys.argv[2]
s = open(p).read()
old = ('        if self.sliding_window is not None:\n'
       '            assert not vllm_config.model_config.use_mla, (\n'
       '                "MLA is not supported for slidingwindow"\n'
       '            )\n')
assert s.count(old) == 1, "anchor found %d times" % s.count(old)
new = ('        if self.sliding_window is not None:\n'
       '            # ' + mark + ': model_config.use_mla describes the TARGET model,\n'
       '            # not this layer. This method is on the non-MLA Attention class, so a\n'
       '            # sliding-window layer here is genuinely non-MLA -- it belongs to the\n'
       '            # DFlash2 drafter, which registers 6 plain SWA layers under an MLA\n'
       '            # target. The original assert predates draft models and rejects that\n'
       '            # legitimate combination. MLA target layers never reach this line.\n')
s = s.replace(old, new, 1)
ast.parse(s)
open(p, "w").write(s)
print("patched")
PYEOF
fi

if ! grep -q "$MARK" "$LAUNCH"; then
  python3 - "$LAUNCH" "$MARK" << 'PYEOF'
import sys
p, mark = sys.argv[1], sys.argv[2]
s = open(p).read()
anchor = 'docker rm -f "$NAME" 2>/dev/null'
assert s.count(anchor) == 1, "anchor %d" % s.count(anchor)
blk = ('# ' + mark + ': allow the DFlash2 drafter\'s sliding-window layers under an MLA target.\n'
       'SWA_PATCH="$HOME/patches-port/attention.py"\n'
       'PORT_MOUNT=()\n'
       'if [ -f "$SWA_PATCH" ]; then\n'
       '  PORT_MOUNT=(-v "$SWA_PATCH:/usr/local/lib/python3.12/dist-packages/vllm/'
       'model_executor/layers/attention/attention.py:ro")\n'
       'fi\n\n')
s = s.replace(anchor, blk + anchor, 1)
old = 'docker run -d --name "$NAME" \\n'
assert s.count(old) == 1, "docker run anchor %d" % s.count(old)
s = s.replace(old, old + '  "${PORT_MOUNT[@]}" \\n', 1)
open(p, "w").write(s)
print("launcher wired")
PYEOF
fi
bash -n "$LAUNCH" || { echo "SYNTAX FAIL"; exit 1; }
echo "$(hostname) patch=$(grep -c "$MARK" "$OUT") launcher=$(grep -c "$MARK" "$LAUNCH")"
