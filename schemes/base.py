from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logs import logger


class BaseScheme(ABC):
    """
    Common scheme lifecycle:

    - train(): default outer-loop scaffold over epochs/batches, calls train_on_batch()
    - save_model(): persist scheme state to self.scheme_file
    - load(): restore scheme state from disk
    - prep_test(): switch to eval mode (e.g., enable usage tracking)
    - inference(): single-example inference returning (answer, cost_usd)
    """

    def __init__(self, args: Any):
        self.args = args

        self.output_subdir = Path(args.output_dir) / f"{args.scheme}_{args.benchmark}"
        self.output_subdir.mkdir(parents=True, exist_ok=True)

        # Always persist scheme artifacts with a *.py suffix.
        # (Even if the artifact is "just a prompt", we still store it in a .py file.)
        self.scheme_file = (self.output_subdir / "prompt").with_name("code.py")
        self.result_file = self.output_subdir / "score.csv"

    # ----------------------------
    # Optional hooks
    # ----------------------------

    def prep_train(self) -> None:
        """Optional hook called before training starts and after each test evaluation."""
        return

    def prep_test(self) -> None:
        """Optional hook called before evaluation / baseline runs."""
        return

    async def train_on_batch(self, batch: List[dict], train_benchmark: Any) -> Dict[str, Any]:
        """
        Optional: inner-loop optimization step.

        Schemes that want to use BaseScheme.train() should override this.
        Schemes that implement their own train() can ignore this method.
        """
        raise NotImplementedError("train_on_batch is not implemented for this scheme.")

    def save_model(self, epoch: Optional[int] = None) -> None:
        """Optional: persist scheme state to self.scheme_file."""
        return

    # ----------------------------
    # Required API
    # ----------------------------

    @abstractmethod
    def load(self, path: Path) -> bool:
        """Load scheme state from disk. Return True if loaded."""
        raise NotImplementedError

    @abstractmethod
    async def inference(self, input_text: str) -> Tuple[str, float]:
        """Return (prediction, cost_usd)."""
        raise NotImplementedError

    # ----------------------------
    # Default training scaffold
    # ----------------------------

    async def train(
        self,
        train_benchmark: Any,
        train_indices: Sequence[int],
        test_benchmark: Optional[Any] = None,
        test_indices: Optional[Sequence[int]] = None,
        test_freq: int = 1,
    ) -> None:
        """
        Default outer-loop training scaffold.

        - Outer loop: epochs over training data.
        - Inner loop: calls train_on_batch(batch, train_benchmark).
        - Optional test loop: run test_benchmark every test_freq epochs.

        Subclasses customize behavior by overriding:
          - prep_train / prep_test
          - train_on_batch
          - save_model / load
        """
        self.prep_train()

        train_data = await train_benchmark.load_data(list(train_indices))

        epochs = max(1, int(getattr(self.args, "epochs", 1)))
        batch_size = max(1, int(getattr(self.args, "batch_size", 1)))
        test_freq = max(1, int(test_freq))

        logger.info(
            f"[train] scheme={getattr(self.args, 'scheme', '')} "
            f"benchmark={getattr(self.args, 'benchmark', '')} "
            f"epochs={epochs} batch_size={batch_size} n_train={len(train_data)} test_freq={test_freq}"
        )

        for epoch in range(1, epochs + 1):
            if len(train_data) > 1:
                random.shuffle(train_data)

            for start in range(0, len(train_data), batch_size):
                batch = train_data[start : start + batch_size]
                metrics = await self.train_on_batch(batch, train_benchmark)
                if metrics:
                    logger.info(f"[train] epoch={epoch} step={start//batch_size} metrics={metrics}")

            # Persist after each epoch (so interrupted runs still leave an artifact).
            self.save_model(epoch=epoch)

            # Periodic test evaluation.
            if (
                test_benchmark is not None
                and test_indices is not None
                and len(test_indices) > 0
                and (epoch % test_freq == 0)
            ):
                logger.info(f"[test] epoch={epoch} n_test={len(test_indices)}")
                self.prep_test()

                await test_benchmark.run_baseline(
                    agent=self.inference,
                    specific_indices=list(test_indices),
                    max_concurrent_tasks=getattr(test_benchmark, "max_concurrent_tasks", 50),
                )

                self.prep_train()

        # Ensure there's always a materialized artifact file after train().
        if not self.scheme_file.exists():
            logger.warning(f"[train] {self.scheme_file} not created; writing placeholder.")
            self.scheme_file.write_text("# Placeholder scheme artifact\n", encoding="utf-8")
