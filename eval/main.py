"""
Main Execution Script for Control-Plane Integrity Evaluation

Usage:
    python -m eval.main quick                             # 3-attack quick test
    python -m eval.main experiment malicious_skills       # single experiment
    python -m eval.main comprehensive gpt-4o-mini 9      # full evaluation
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .evaluator import ControlPlaneEvaluator
from .attack_generator import AttackGenerator

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "evaluation.log"),
        ],
    )
    logger.info("Logging configured")


def check_environment() -> bool:
    required_vars = ["OPENAI_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]

    if missing:
        logger.error("Missing required environment variables: %s", missing)
        logger.info("Please set these variables in .env file or environment")
        return False

    logger.info("Environment check passed")
    return True


def run_quick_test() -> dict:
    logger.info("=" * 80)
    logger.info("RUNNING QUICK TEST")
    logger.info("=" * 80)

    evaluator = ControlPlaneEvaluator()
    results = evaluator.run_malicious_skills_experiment(
        model_name="gpt-4o-mini", num_attacks=3,
    )

    logger.info("Quick test completed")
    logger.info("Protected ASR: %.2f%%", results["protected_agent"]["attack_success_rate"] * 100)
    logger.info("Baseline ASR: %.2f%%", results["baseline_agent"]["attack_success_rate"] * 100)
    return results


def run_single_experiment(
    experiment_name: str,
    model_name: str = "gpt-4o-mini",
    num_attacks: int = 9,
) -> dict | None:
    logger.info("=" * 80)
    logger.info("RUNNING EXPERIMENT: %s", experiment_name)
    logger.info("=" * 80)

    evaluator = ControlPlaneEvaluator()

    dispatch = {
        "malicious_skills": evaluator.run_malicious_skills_experiment,
        "tool_poisoning": evaluator.run_tool_poisoning_experiment,
        "memory_poisoning": evaluator.run_memory_poisoning_experiment,
        "indirect_injection": evaluator.run_indirect_injection_experiment,
    }

    func = dispatch.get(experiment_name)
    if func is None:
        logger.error("Unknown experiment: %s", experiment_name)
        return None

    results = func(model_name, num_attacks)

    logger.info("Experiment %s completed", experiment_name)
    logger.info("Protected ASR: %.2f%%", results["protected_agent"]["attack_success_rate"] * 100)
    logger.info("Baseline ASR: %.2f%%", results["baseline_agent"]["attack_success_rate"] * 100)
    logger.info("Improvement: %.1f%%", results["comparison"]["asr_reduction_percentage"])
    logger.info("Statistically significant: %s", results["comparison"]["statistically_significant"])
    logger.info("Theorem validated: %s", results["comparison"]["theorem_validated"])
    return results


def run_comprehensive_evaluation(
    models: list[str] | None = None,
    attacks_per_experiment: int = 9,
) -> dict:
    if models is None:
        models = ["gpt-4o-mini"]

    logger.info("=" * 80)
    logger.info("RUNNING COMPREHENSIVE EVALUATION")
    logger.info("=" * 80)
    logger.info("Models: %s", models)
    logger.info("Attacks per experiment: %d", attacks_per_experiment)

    evaluator = ControlPlaneEvaluator()
    results = evaluator.run_comprehensive_evaluation(
        models=models, attacks_per_experiment=attacks_per_experiment,
    )

    summary = results["summary"]
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("Average Protected ASR: %.2f%%", summary["avg_protected_asr"] * 100)
    logger.info("Average Baseline ASR: %.2f%%", summary["avg_baseline_asr"] * 100)
    logger.info("Average Improvement: %.2f%%", summary["avg_improvement"] * 100)
    logger.info("Theorem Validation:")
    tv = summary["theorem_validation"]
    logger.info("  Protected ASR < 5%%: %s", tv["protected_asr_below_5pct"])
    logger.info("  Baseline ASR > 50%%: %s", tv["baseline_asr_above_50pct"])
    logger.info("  Significant Improvement: %s", tv["significant_improvement"])
    logger.info("  THEOREM VALIDATED: %s", tv["theorem_validated"])

    csv_path = evaluator.export_results_to_csv()
    logger.info("Results exported to: %s", csv_path)
    return results


def show_attack_statistics() -> None:
    logger.info("=" * 80)
    logger.info("ATTACK STATISTICS")
    logger.info("=" * 80)

    generator = AttackGenerator()
    stats = generator.get_attack_statistics()

    logger.info("Total attacks available: %d", stats["total"])
    logger.info("By type:")
    for attack_type, count in stats["by_type"].items():
        logger.info("  %s: %d", attack_type, count)
    logger.info("By obfuscation level:")
    for level, count in stats["by_obfuscation"].items():
        logger.info("  Level %s: %d", level, count)


def main() -> None:
    load_dotenv()
    setup_logging()

    logger.info("=" * 80)
    logger.info("CONTROL-PLANE INTEGRITY THEOREM EVALUATION")
    logger.info("=" * 80)

    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        return

    show_attack_statistics()

    args = sys.argv[1:]
    if not args:
        logger.info("No command specified, running comprehensive evaluation")
        run_comprehensive_evaluation()
        return

    command = args[0]

    if command == "quick":
        run_quick_test()

    elif command == "experiment":
        if len(args) < 2:
            logger.error("Usage: python -m eval.main experiment <name> [model] [num_attacks]")
            logger.info(
                "Available experiments: malicious_skills, tool_poisoning, "
                "memory_poisoning, indirect_injection"
            )
            return
        exp_name = args[1]
        model = args[2] if len(args) > 2 else "gpt-4o-mini"
        num_attacks = int(args[3]) if len(args) > 3 else 9
        run_single_experiment(exp_name, model, num_attacks)

    elif command == "comprehensive":
        models = args[1].split(",") if len(args) > 1 else ["gpt-4o-mini"]
        num_attacks = int(args[2]) if len(args) > 2 else 9
        run_comprehensive_evaluation(models, num_attacks)

    else:
        logger.error("Unknown command: %s", command)
        logger.info("Available commands: quick, experiment, comprehensive")

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
