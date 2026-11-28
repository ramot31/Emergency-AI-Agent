"""Évaluation synthétique distincte de la comparaison utilisée pour la politique."""
from __future__ import annotations

import json

from .compare_policies import compare
from .config import DEFAULT_ARTIFACT_DIR, PACKAGE_DIR
from .data_generation import generate_dataset

DATA = PACKAGE_DIR / "data" / "policy_holdout_synthetic.csv"
RESULT = DEFAULT_ARTIFACT_DIR / "policy_holdout_comparison.json"
SEED = 734921


def main() -> None:
    if not DATA.exists():
        generate_dataset(DATA, n_samples=1200, random_state=SEED)
    result = compare(DATA, RESULT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
