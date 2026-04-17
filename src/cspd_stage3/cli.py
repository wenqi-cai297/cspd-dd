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
        help="Encode dataset images to DINOv2 features for clustering (Stage 3A)",
    )
    encode_parser.add_argument("--dataset-root", required=True, help="ImageFolder-style dataset root")
    encode_parser.add_argument("--render-input", required=True, help="Stage 1C render records.jsonl path")
    encode_parser.add_argument("--output-dir", required=True, help="Directory for encoded tensor outputs")
    encode_parser.add_argument("--resolution", type=int, default=512, help="Image resolution for loading")
    encode_parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    encode_parser.add_argument("--device", default="cuda", help="Torch device")
    encode_parser.add_argument("--encode-vae", action="store_true", help="Also encode VAE latents (needed for mode guidance in Stage 4)")
    encode_parser.add_argument("--vae-model-name", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model for VAE loading")

    # --- cluster ---
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster latents per class and extract visual/semantic modes (Stage 3B+3C)",
    )
    cluster_parser.add_argument("--encode-dir", required=True, help="Directory with Stage 3A encode outputs")
    cluster_parser.add_argument("--output-dir", required=True, help="Directory for mode outputs")
    cluster_parser.add_argument("--ipc", type=int, required=True, help="Images per class (number of clusters per class)")
    cluster_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    cluster_parser.add_argument("--cluster-method", default="kmeans", choices=["kmeans", "hdbscan"], help="Clustering method: kmeans (baseline) or hdbscan (mode discovery)")
    cluster_parser.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min_cluster_size (ignored for kmeans)")
    cluster_parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN min_samples: core point neighborhood density (ignored for kmeans)")
    cluster_parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimensions for HDBSCAN pre-processing (ignored for kmeans)")
    cluster_parser.add_argument("--diversify-captions", action="store_true", help="Replace medoid captions with most diverse alternatives (experimental, may hurt accuracy)")

    # --- run (encode + cluster in one shot) ---
    run_parser = subparsers.add_parser(
        "run",
        help="Full Stage 3 pipeline: encode + cluster in one shot",
    )
    run_parser.add_argument("--dataset-root", required=True, help="ImageFolder-style dataset root")
    run_parser.add_argument("--render-input", required=True, help="Stage 1C render records.jsonl path")
    run_parser.add_argument("--output-dir", required=True, help="Root output directory for Stage 3")
    run_parser.add_argument("--ipc", type=int, required=True, help="Images per class")
    run_parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    run_parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    run_parser.add_argument("--device", default="cuda", help="Torch device")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--cluster-method", default="kmeans", choices=["kmeans", "hdbscan"], help="Clustering method: kmeans (baseline) or hdbscan (mode discovery)")
    run_parser.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min_cluster_size (ignored for kmeans)")
    run_parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN min_samples: core point neighborhood density (ignored for kmeans)")
    run_parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimensions for HDBSCAN pre-processing (ignored for kmeans)")
    run_parser.add_argument("--encode-vae", action="store_true", help="Also encode VAE latents (needed for mode guidance in Stage 4)")
    run_parser.add_argument("--vae-model-name", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model for VAE loading")
    run_parser.add_argument("--diversify-captions", action="store_true", help="Replace medoid captions with most diverse alternatives (experimental)")

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
            resolution=args.resolution,
            batch_size=args.batch_size,
            device=args.device,
            encode_vae=args.encode_vae,
            vae_model_name=args.vae_model_name,
        )
        print(json.dumps({
            "dino_embeds_path": result.dino_embeds_path,
            "vae_latents_path": result.vae_latents_path,
            "num_samples": result.num_samples,
            "dino_embed_dim": result.dino_embed_dim,
        }, indent=2))

    elif args.command == "cluster":
        from cspd_stage3.cluster import run_stage3_clustering

        result = run_stage3_clustering(
            encode_dir=args.encode_dir,
            output_dir=args.output_dir,
            ipc=args.ipc,
            seed=args.seed,
            method=args.cluster_method,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            pca_dim=args.pca_dim,
            diversify_captions=args.diversify_captions,
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
            resolution=args.resolution,
            batch_size=args.batch_size,
            device=args.device,
            encode_vae=args.encode_vae,
            vae_model_name=args.vae_model_name,
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
            method=args.cluster_method,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            pca_dim=args.pca_dim,
            diversify_captions=args.diversify_captions,
        )

        print()
        print("=" * 60)
        print(f"[Stage 3] Complete: {cluster_result.total_modes} modes "
              f"({cluster_result.num_classes} classes × IPC={args.ipc})")
        print(f"[Stage 3] Modes index: {cluster_result.modes_index_path}")
        print("=" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
