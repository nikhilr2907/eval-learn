from pathlib import Path
from typing import Dict, Optional, Tuple


def load_vocab(vocab_dir: Optional[str]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load vocab files from the given directory path.

    The directory must contain:
        words.txt  — tab-separated  id <TAB> label(s)
        gloss.txt  — tab-separated  id <TAB> definition
        is_a.txt   — space-separated parent child pairs (optional)

    Returns
    -------
    class_map        : {id -> label string}          from words.txt
    gloss_map        : {id -> definition string}     from gloss.txt
    ancestors_map    : {id -> set of ancestor ids}   from is_a.txt
    descendants_map  : {id -> set of descendant ids} from is_a.txt
    """
    if not vocab_dir:
        raise ValueError(
            "vocab_dir must be a path to a directory containing "
            "wnid.txt, words.txt, gloss.txt, and is_a.txt"
        )
    base = Path(vocab_dir)

    class_map: Dict[str, str] = {}
    gloss_map: Dict[str, str] = {}
    ancestors_map: Dict[str, set] = {}
    descendants_map: Dict[str, set] = {}

    with open(base / "words.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                class_map[parts[0]] = parts[1]

    with open(base / "gloss.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                gloss_map[parts[0]] = parts[1]

    is_a_path = base / "is_a.txt"
    if is_a_path.exists():
        with open(is_a_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    parent, child = parts
                    ancestors_map.setdefault(child, set()).add(parent)
                    descendants_map.setdefault(parent, set()).add(child)

    return class_map, gloss_map, ancestors_map, descendants_map


def load_wnids(vocab_dir: Optional[str]) -> list:
    """Return ordered list of IDs from wnid.txt in the given vocab_dir."""
    if not vocab_dir:
        raise ValueError("vocab_dir must be specified to load wnid.txt")
    with open(Path(vocab_dir) / "wnid.txt") as f:
        return [line.strip() for line in f if line.strip()]
