"""CLI entrypoint for CSPD Stage 3 — visual/semantic mode discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSPD Stage 3 CLI for visual/semantic mode discovery via latent clustering"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- encode ---
    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode dataset images to VAE latents and captions to text embeddings (Stage 3A)",
    )
    encode_parser.add_argument("--dataset-root", required=True, help="ImageFolder-style dataset root")
    encode_parser.add_argument("--render-input", required=True, help="Stage 1C render records.jsonl path")
    encode_parser.add_argument("--output-dir", required=True, help="Directory for encoded tensor outputs")
    encode_parser.add_argument("--model-name", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model identifier")
    encode_parser.add_argument("--resolution", type=int, default=512, help="Image resolution for VAE encoding")
    encode_parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    encode_parser.add_argument("--device", default="cuda", help="Torch device")
    encode_parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Weight dtype")

    # --- cluster ---
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster latents per class and extract visual/semantic modes (Stage 3B+3C)",
    )
    cluster_parser.add_argument("--encode-dir", required=True, help="Directory with Stage 3A encode outputs")
    cluster_parser.add_argument("--output-dir", required=True, help="Directory for mode outputs")
    cluster_parser.add_argument("--ipc", type=int, required=True, help="Images per class (number of clusters per class)")
    cluster_parser.add_argument("--seed", type=int, default=42, help="Random seed for K-Means")

    # --- run (encode + cluster in one shot) ---
    run_parser = subparsers.add_parser(
        "run",
        help="Full Stage 3 pipeline: encode + cluster in one shot",
    )
    run_parser.add_argument("--dataset-root", required=True, help="ImageFolder-style dataset root")
    run_parser.add_argument("--render-input", required=True, help="Stage 1C render records.jsonl path")
    run_parser.add_argument("--output-dir", required=True, help="Root output directory for Stage 3")
    run_parser.add_argument("--ipc", type=int, required=True, help="Images per class")
    run_parser.add_argument("--model-name", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model identifier")
    run_parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    run_parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    run_parser.add_argument("--device", default="cuda", help="Torch device")
    run_parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Weight dtype")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "encode":
        from cspd_stage3.encode import encode_dataset

        result = encode_dataset(
            dataset_root=args.dataset_root,
            render_input=args.render_input,
            output_dir=args.output_dir,
            model_name=args.model_name,
            resolution=args.resolution,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
        )
        print(json.dumps({
            "latents_path": result.latents_path,
            "text_embeds_path": result.text_embeds_path,
            "num_samples": result.num_samples,
            "latent_shape": result.latent_shape,
        }, indent=2))

    elif args.command == "cluster":
        from cspd_stage3.cluster import run_stage3_clustering

        result = run_stage3_clustering(
            encode_dir=args.encode_dir,
            output_dir=args.output_dir,
            ipc=args.ipc,
            seed=args.seed,
        )
        print(json.dumps({
            "output_dir": result.output_dir,
            "ipc": result.ipc,
            "num_classes": result.num_classes,
            "total_modes": result.total_modes,
        }, indent=2))

    elif args.command == "run":
        from cspd_stage3.encode import encode_dataset
        from cspd_stage3.cluster import run_stage3_clustering

        output_dir = Path(args.output_dir)
        encode_dir = output_dir / "encoded"
        modes_dir = output_dir / "modes"

        print("=" * 60)
        print("[Stage 3] Phase 1: Encoding")
        print("=" * 60)
        encode_result = encode_dataset(
            dataset_root=args.dataset_root,
            render_input=args.render_input,
            output_dir=str(encode_dir),
            model_name=args.model_name,
            resolution=args.resolution,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
        )

        print()
        print("=" * 60)
        print("[Stage 3] Phase 2: Clustering + Mode Extraction")
        print("=" * 60)
        cluster_result = run_stage3_clustering(
            encode_dir=str(encode_dir),
            output_dir=str(modes_dir),
            ipc=args.ipc,
            seed=args.seed,
        )

        print()
        print("=" * 60)
        print(f"[Stage 3] Complete: {cluster_result.total_modes} modes "
              f"({cluster_result.num_classes} classes × IPC={args.ipc})")
        print(f"[Stage 3] Visual modes:   {cluster_result.visual_modes_path}")
        print(f"[Stage 3] Semantic modes:  {cluster_result.semantic_modes_path}")
        print(f"[Stage 3] Modes index:     {cluster_result.modes_index_path}")
        print("=" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
