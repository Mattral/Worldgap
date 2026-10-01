#!/usr/bin/env bash
# Downloads HaGRID and EgoHands into data/raw/.
#
# READ FIRST -- you probably do not need this for a V1 run.
#
# HaGRID is an IMAGE dataset and cannot supply V1's trajectories; see
# docs/temporal_provenance.md. V1's primary real source is video, including
# your own webcam recordings, which need no download at all --
# docs/v1_real_data_runbook.md walks through that in under an hour.
#
# This script remains useful for: EgoHands (video-derived frames, a genuine
# second source), and HaGRID for the frame-level landmark distribution
# comparison that `extract_static_pose_rollouts()` supports.
#
# NOT run or verified in the sandbox that produced this repo: its network
# allow-list doesn't reach Kaggle or the EgoHands host. Run this on your own
# machine, and note it in CHANGELOG.md if the URLs below have moved.
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

echo "Reminder: HaGRID stills cannot be assembled into trajectories --"
echo "hagrid.extract_rollout_from_frames() raises on purpose. Use"
echo "extract_static_pose_rollouts() for a frame-level comparison, or record"
echo "video (docs/v1_real_data_runbook.md) for the real V1 run."
