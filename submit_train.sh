#!/bin/bash
# Submit one job per (setting x method) for all 3 methods:
#   simple            -> simple prompt,       no personalization probe, no reward shaping
#   pers_no_reward    -> personalized prompt, probe ON,                 no reward shaping
#   pers_with_reward  -> personalized prompt, probe ON,                 +2x positive reward when personalized
#
# Violation probe is always ON; flagged conversations get reward = -6 (see
# src/probes/violation_probe.py::VIOLATION_PENALTY).
#
# Outputs live at:
#   outputs/<method>/<setting>/<category>/<seller_prompt_mode>/<random_init|buyer_init>/
#
# Usage: ./submit_train.sh

set -euo pipefail

SETTINGS=("likelihood" "preference")
CATEGORIES=("movies")
RANDOM_INITIAL_VOTES=("True")

# method | seller_prompt_mode | personalization_probe | personalization_reward
METHODS=(
    "simple|simple|False|False"
    "pers_no_reward|personalized|True|False"
    "pers_with_reward|personalized|True|True"
)

TOTAL=$((${#SETTINGS[@]} * ${#CATEGORIES[@]} * ${#RANDOM_INITIAL_VOTES[@]} * ${#METHODS[@]}))

echo "Submitting ${TOTAL} jobs"
echo "  Settings:   ${SETTINGS[*]}"
echo "  Categories: ${CATEGORIES[*]}"
echo "  Random initial vote: ${RANDOM_INITIAL_VOTES[*]}"
echo "  Methods:    simple, pers_no_reward, pers_with_reward"
echo ""

COUNTER=1

for setting in "${SETTINGS[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        for riv in "${RANDOM_INITIAL_VOTES[@]}"; do
            for spec in "${METHODS[@]}"; do
                IFS='|' read -r method spm pp pr <<< "$spec"
                echo "[$COUNTER/$TOTAL] method=${method} setting=${setting} category=${category} random_initial_vote=${riv}"
                sbatch --export=setting="$setting",dataset_category="$category",random_initial_vote="$riv",seller_prompt_mode="$spm",personalization_probe="$pp",personalization_reward="$pr",output_dir="outputs" \
                    run_train.sh
                echo ""
                COUNTER=$((COUNTER + 1))
            done
        done
    done
done

echo "Submitted all ${TOTAL} jobs."
