"""Optional video metrics and checkpoint checks; heavy runtimes load on demand."""
from pathlib import Path


WEIGHTS = {
    'clipiqa': (
        'pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth',
        'clip/RN50.pt',
    ),
    'motion': ('amt_model/amt-s.pth',),
}


def check_weights(metric: str, cache: Path) -> None:
    for relative in WEIGHTS[metric]:
        path = cache / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(
                f'Missing checkpoint: {path}. '
                'See docs/metrics.md#checkpoint-preparation for download instructions.'
            )
