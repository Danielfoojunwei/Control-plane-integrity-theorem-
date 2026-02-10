"""
Dataset Downloader for Real Benchmarks

Downloads canonical datasets from HuggingFace and GitHub:
- Agent-SafetyBench (Zhang et al., 2024)
- INJECAGENT (Zhan et al., 2024)
- PINT Benchmark (Lakera)
- NVIDIA Aegis AI Content Safety Dataset
- Agent Security Bench (ASB)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and prepare evaluation datasets."""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dataset directory: %s", self.data_dir)

    def download_agent_safetybench(self) -> bool:
        """Download Agent-SafetyBench dataset.
        Repository: https://github.com/thu-coai/Agent-SafetyBench
        """
        logger.info("Downloading Agent-SafetyBench...")
        try:
            output_dir = self.data_dir / "agent_safetybench"
            output_dir.mkdir(exist_ok=True)
            if not (output_dir / ".git").exists():
                subprocess.run(
                    ["git", "clone", "https://github.com/thu-coai/Agent-SafetyBench.git",
                     str(output_dir)],
                    check=True,
                )
                logger.info("Agent-SafetyBench cloned successfully")
            else:
                logger.info("Agent-SafetyBench already exists, pulling updates")
                subprocess.run(["git", "pull"], cwd=output_dir, check=True)
            return True
        except Exception as e:
            logger.error("Failed to download Agent-SafetyBench: %s", e)
            return False

    def download_injecagent(self) -> bool:
        """Download INJECAGENT benchmark.
        Repository: https://github.com/uiuc-kang-lab/InjecAgent
        """
        logger.info("Downloading INJECAGENT...")
        try:
            output_dir = self.data_dir / "injecagent"
            output_dir.mkdir(exist_ok=True)
            if not (output_dir / ".git").exists():
                subprocess.run(
                    ["git", "clone", "https://github.com/uiuc-kang-lab/InjecAgent.git",
                     str(output_dir)],
                    check=True,
                )
                logger.info("INJECAGENT cloned successfully")
            else:
                logger.info("INJECAGENT already exists, pulling updates")
                subprocess.run(["git", "pull"], cwd=output_dir, check=True)
            return True
        except Exception as e:
            logger.error("Failed to download INJECAGENT: %s", e)
            return False

    def download_pint_benchmark(self) -> bool:
        """Download PINT Benchmark (Lakera).
        Repository: https://github.com/lakeraai/pint-benchmark
        """
        logger.info("Downloading PINT Benchmark...")
        try:
            output_dir = self.data_dir / "pint_benchmark"
            output_dir.mkdir(exist_ok=True)
            if not (output_dir / ".git").exists():
                subprocess.run(
                    ["git", "clone", "https://github.com/lakeraai/pint-benchmark.git",
                     str(output_dir)],
                    check=True,
                )
                logger.info("PINT Benchmark cloned successfully")
            else:
                logger.info("PINT Benchmark already exists, pulling updates")
                subprocess.run(["git", "pull"], cwd=output_dir, check=True)
            return True
        except Exception as e:
            logger.error("Failed to download PINT Benchmark: %s", e)
            return False

    def download_nvidia_aegis(self) -> bool:
        """Download NVIDIA Aegis AI Content Safety Dataset from HuggingFace.
        Dataset: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
        """
        logger.info("Downloading NVIDIA Aegis AI Content Safety Dataset...")
        try:
            from huggingface_hub import snapshot_download

            output_dir = self.data_dir / "nvidia_aegis"
            output_dir.mkdir(exist_ok=True)
            snapshot_download(
                repo_id="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                repo_type="dataset",
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("NVIDIA Aegis dataset downloaded successfully")
            return True
        except Exception as e:
            logger.error("Failed to download NVIDIA Aegis dataset: %s", e)
            return False

    def download_agent_security_bench(self) -> bool:
        """Download Agent Security Bench (ASB).
        Repository: https://github.com/agiresearch/ASB
        """
        logger.info("Downloading Agent Security Bench...")
        try:
            output_dir = self.data_dir / "agent_security_bench"
            output_dir.mkdir(exist_ok=True)
            if not (output_dir / ".git").exists():
                subprocess.run(
                    ["git", "clone", "https://github.com/agiresearch/ASB.git",
                     str(output_dir)],
                    check=True,
                )
                logger.info("Agent Security Bench cloned successfully")
            else:
                logger.info("Agent Security Bench already exists, pulling updates")
                subprocess.run(["git", "pull"], cwd=output_dir, check=True)
            return True
        except Exception as e:
            logger.error("Failed to download Agent Security Bench: %s", e)
            return False

    def download_all(self) -> Dict[str, bool]:
        """Download all datasets."""
        logger.info("=" * 80)
        logger.info("DOWNLOADING ALL DATASETS")
        logger.info("=" * 80)

        results = {
            "agent_safetybench": self.download_agent_safetybench(),
            "injecagent": self.download_injecagent(),
            "pint_benchmark": self.download_pint_benchmark(),
            "nvidia_aegis": self.download_nvidia_aegis(),
            "agent_security_bench": self.download_agent_security_bench(),
        }

        logger.info("=" * 80)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        for dataset, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info("%s: %s", dataset, status)
        total = len(results)
        successful = sum(results.values())
        logger.info("Total: %d/%d datasets downloaded successfully", successful, total)
        return results

    def verify_datasets(self) -> Dict[str, bool]:
        """Verify that datasets are properly downloaded."""
        logger.info("Verifying datasets...")
        verification = {
            "agent_safetybench": (self.data_dir / "agent_safetybench").exists(),
            "injecagent": (self.data_dir / "injecagent").exists(),
            "pint_benchmark": (self.data_dir / "pint_benchmark").exists(),
            "nvidia_aegis": (self.data_dir / "nvidia_aegis").exists(),
            "agent_security_bench": (self.data_dir / "agent_security_bench").exists(),
        }
        for dataset, exists in verification.items():
            status = "FOUND" if exists else "MISSING"
            logger.info("%s: %s", dataset, status)
        return verification

    def get_dataset_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about downloaded datasets."""
        info: Dict[str, Dict[str, str]] = {}

        datasets = {
            "agent_safetybench": {
                "description": "349 environments, 2000 test cases, 8 safety categories",
                "citation": "Zhang et al., 2024, arXiv:2412.14470",
            },
            "injecagent": {
                "description": "1054 test cases, 17 user tools, 62 attacker tools",
                "citation": "Zhan et al., 2024, arXiv:2403.02691",
            },
            "pint_benchmark": {
                "description": "Prompt injection detection benchmark by Lakera",
                "citation": "Lakera, 2024",
            },
            "nvidia_aegis": {
                "description": "25,007 prompts for content safety evaluation",
                "citation": "NVIDIA, 2024",
            },
            "agent_security_bench": {
                "description": "Benchmark for agent security attacks and defenses",
                "citation": "arXiv:2410.02644",
            },
        }

        for name, meta in datasets.items():
            path = self.data_dir / name
            if path.exists():
                info[name] = {
                    "path": str(path),
                    "description": meta["description"],
                    "citation": meta["citation"],
                }

        return info


def main():
    """Main entry point for dataset downloader."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    downloader = DatasetDownloader()
    downloader.download_all()
    downloader.verify_datasets()
    info = downloader.get_dataset_info()
    logger.info("=" * 80)
    logger.info("DATASET INFORMATION")
    logger.info("=" * 80)
    for dataset, details in info.items():
        logger.info("%s:", dataset)
        logger.info("  Path: %s", details["path"])
        logger.info("  Description: %s", details["description"])
        logger.info("  Citation: %s", details["citation"])


if __name__ == "__main__":
    main()
