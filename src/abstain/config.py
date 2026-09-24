import yaml


def load_config(path: str = "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    """Load YAML config; overrides like ['llm.model=Qwen/Qwen2.5-1.5B-Instruct', 'retrieval.final_k=3']."""
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    for o in overrides or []:
        key, val = o.split("=", 1)
        d = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    return cfg
