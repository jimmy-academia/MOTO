from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logs import logger

from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple



class BaseScheme(ABC):
    """
    Common scheme lifecycle:

    - train(): default outer-loop scaffold over epochs/batches, calls train_one_batch()
    - save_model(): persist scheme state to self.scheme_file
    - load(): restore scheme state from disk
    - prep_test(): switch to eval mode (e.g., enable usage tracking)
    - inference(): single-example inference returning (answer, cost_usd)
    """

    def __init__(self, args: Any):
        self.args = args

        self.output_subdir = Path(args.output_dir) / f"{args.scheme}_{args.benchmark}"
        self.output_subdir.mkdir(parents=True, exist_ok=True)
        self.scheme_file = self.output_subdir / "scheme.py"
        self.result_file = self.output_subdir / "result.csv"

    # ----------------------------
    # Optional hooks
    # ----------------------------

    def prep_train(self) -> None:
        """Optional hook called before training starts and after each test evaluation."""
        return

    def prep_test(self) -> None:
        """Optional hook called before evaluation / baseline runs."""
        return

    def iter_batches(self, data: List[dict], batch_size: int, keys: List[str]) -> Iterator[Any]:
        xs = list(data)
        random.shuffle(xs)
        def unpack_batch(batch, keys):
            return tuple([ex[k] for ex in batch] for k in keys)

        if self.args.batch_mode == "sample":
            qk, ak = keys
            for i in range(0, len(xs), batch_size):
                batch = xs[i : i + batch_size]
                yield unpack_batch(batch, (qk, ak))
            return
        if self.args.batch_mode == "meta":
            ck, qk, ak = keys
            for i in range(0, len(xs), batch_size):
                batch = xs[i : i + batch_size]
                yield unpack_batch(batch, (ck, qk, ak))
            return

    async def train_one_batch(self, batch: List[dict], calculate_score: Any) -> Dict[str, Any]:
        """
        Optional: inner-loop optimization step.
        Schemes that implement their own train() can ignore this method.
        """
        raise NotImplementedError("train_one_batch is not implemented for this scheme.")

    @abstractmethod
    def save_model(self, epoch: Optional[int] = None) -> None:
        """Persist scheme state to self.scheme_file."""
        return

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
        - Inner loop: calls train_one_batch(batch, train_benchmark).
        - Optional test loop: run test_benchmark every test_freq epochs.

        Subclasses customize behavior by overriding:
          - prep_train / prep_test
          - train_one_batch
          - save_model / load
        """
        self.prep_train()

        data = await train_benchmark.load_data(train_indices)
        keys = [train_benchmark.q_key, train_benchmark.a_key]
        if self.args.batch_mode == 'meta':
            keys = [train_benchmark.c_key] + keys

        epochs = max(1, int(getattr(self.args, "epochs", 1)))
        batch_size = max(1, int(getattr(self.args, "batch_size", 1)))
        test_freq = max(1, int(test_freq))
        total_steps = (len(data) + batch_size - 1) // batch_size


        logger.info(
            f"[train] scheme={getattr(self.args, 'scheme', '')} "
            f"benchmark={getattr(self.args, 'benchmark', '')} "
            f"epochs={epochs} batch_size={batch_size} n_train={len(data)} test_freq={test_freq}"
        )

        for epoch in range(1, epochs + 1):
            for step, batch in enumerate(self.iter_batches(data, batch_size, keys), start=1):
                metrics = await self.train_one_batch(batch, train_benchmark.calculate_score)
                if metrics:
                    logger.info(f"[train] epoch={epoch} step={step/total_steps} metrics={metrics}")

            self.save_model(epoch=epoch)

            # Periodic test evaluation.
            if (
                test_benchmark is not None and test_indices is not None
                and len(test_indices) > 0 and (epoch % test_freq == 0)
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
        self.save_model()