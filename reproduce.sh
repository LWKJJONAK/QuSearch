#!/bin/sh
# Usage: ./reproduce.sh [data|checks]      (default: data)
#   data     regenerate every file in data/ into output/data and compare the two
#   checks   run the finite-size checks of the Supplemental Material
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

case "${1:-data}" in
    data)
        cargo build --release --offline
        mkdir -p output/data
        target/release/torus spectral > output/data/spectral.csv
        target/release/torus walk > output/data/tulsi_independent.csv
        target/release/marked_equilibrium output/data
        "$PYTHON" python/make_data.py output/data
        "$PYTHON" python/compare.py output/data data
        ;;
    checks)
        cargo build --release --offline
        mkdir -p output/checks
        target/release/check_bound > output/checks/bound.csv
        target/release/check_graphs > output/checks/graphs.txt
        ;;
    *)
        sed -n '2,4p' "$0" >&2
        exit 2
        ;;
esac
echo "Done; see output/."
