#!/usr/bin/env bash
# Downloads HaGRID and EgoHands into data/raw/.
#
# NOT run or verified in the scaffolding session that produced this repo: that
# sandbox's network allow-list doesn't reach Kaggle or the EgoHands host. Run
# this on your own machine (or in Claude Code) where you have normal internet
# access, and open an issue/note in CHANGELOG.md if the URLs below have moved.
#
# Per spec 12.16: raw dataset files MUST NOT be committed to this repo. This
# script is the only sanctioned way to populate data/raw/ — check dataset
# license terms (linked below) before any derived data leaves your machine.

set -euo pipefail

DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$DATA_ROOT"

echo "== HaGRID =="
echo "Hosted on Kaggle: https://www.kaggle.com/datasets/kapitanov/hagrid"
echo "Also linked from: https://github.com/hukenovs/hagrid"
echo "License: check the Kaggle page before any redistribution of derived data (spec 12.16)."
echo "This script does not auto-download HaGRID -- it requires Kaggle API credentials."
echo "  1. pip install kaggle"
echo "  2. place your kaggle.json API token at ~/.kaggle/kaggle.json"
echo "  3. kaggle datasets download -d kapitanov/hagrid -p \"$DATA_ROOT/hagrid\" --unzip"
echo ""

echo "== EgoHands =="
echo "Project page: http://vision.soic.indiana.edu/projects/egohands/"
echo "Direct archive: http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
echo "Citation required (Bambach et al., ICCV 2015) -- see the project page. Check"
echo "current license terms there before any redistribution of derived data."
echo "Original labels are polygon segmentations in a MATLAB format, not the"
echo "landmark format this repo needs -- you will still need MediaPipe over the"
echo "raw frames, same as for HaGRID."
mkdir -p "$DATA_ROOT/egohands"
echo "Fetching EgoHands archive (this WILL fail in the sandbox that generated"
echo "this script, since vision.soic.indiana.edu is not network-reachable from it):"
echo "  curl -L -o \"$DATA_ROOT/egohands/egohands_data.zip\" \\"
echo "    http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
echo "  unzip \"$DATA_ROOT/egohands/egohands_data.zip\" -d \"$DATA_ROOT/egohands\""
echo ""

echo "Once both are in place, run the Phase 0 data audit (ROADMAP.md) before"
echo "building anything on top of them -- confirm canonical-gesture sample"
echo "counts are sufficient (see hagrid.py CANONICAL_GESTURES)."
