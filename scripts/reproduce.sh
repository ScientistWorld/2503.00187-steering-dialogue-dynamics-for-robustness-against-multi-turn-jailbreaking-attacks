#!/bin/bash
set -euo pipefail

cd /home/user

scripts/method.sh
scripts/baseline.sh
scripts/evaluate.sh
