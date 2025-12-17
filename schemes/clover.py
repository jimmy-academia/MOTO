from schemes.base import BaseScheme

class CloverScheme(BaseScheme):
    def __init__(self, args: Any):
        super().__init__(args)

    async def train_one_batch(self, batch: List[dict], calculate_score: Any)

    def save_model(self):
        pass

    def load(self):
        pass