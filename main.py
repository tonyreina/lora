#!/usr/bin/env python3
"""Main entry point for medical LLM training and inference pipeline using Hydra."""

import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the medical LLM pipeline."""
    
    print("🚀 Medical LLM Pipeline")
    print("=" * 60)
    print(f"Mode: {cfg.mode}")
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)
    
    # Set global configuration
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if cfg.mode == "train":
        from pipeline import run_training
        run_training(cfg)
        
    elif cfg.mode == "inference":
        from pipeline import run_inference
        run_inference(cfg)
        
    elif cfg.mode == "demo":
        from pipeline import run_demo
        run_demo(cfg)
        
    elif cfg.mode == "pipeline":
        from pipeline import run_full_pipeline
        run_full_pipeline(cfg)
        
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()