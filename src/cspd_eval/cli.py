"""CLI entrypoint for CSPD evaluation."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSPD Evaluation: train classifiers on distilled datasets and evaluate on real test sets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser(
        "run",
        help="Train a classifier on a distilled dataset and report accuracy on the real validation set",
    )
    eval_parser.add_argument("--distilled-dir", required=True, help="Path to distilled dataset (ImageFolder format)")
    eval_parser.add_argument("--val-dir", required=True, help="Path to real validation dataset (ImageFolder format)")
    eval_parser.add_argument("--arch", default="convnet", choices=["convnet", "resnet18", "resnet_ap"], help="Classifier architecture")
    eval_parser.add_argument("--nclass", type=int, default=10, help="Number of classes")
    eval_parser.add_argument("--ipc", type=int, default=10, help="Images per class in the distilled dataset")
    eval_parser.add_argument("--size", type=int, default=224, help="Image resolution")
    eval_parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    eval_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    eval_parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    eval_parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    eval_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    eval_parser.add_argument("--repeat", type=int, default=3, help="Number of independent training runs to average")
    eval_parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    eval_parser.add_argument("--save-dir", default=None, help="Directory to save evaluation results JSON")
    eval_parser.add_argument("--epochs", type=int, default=None, help="Override auto-computed training epochs")

    # Convenience: run all 3 architectures
    all_parser = subparsers.add_parser(
        "run-all",
        help="Run evaluation with all three architectures (ConvNet-6, ResNet-18, ResNetAP-10)",
    )
    all_parser.add_argument("--distilled-dir", required=True, help="Path to distilled dataset (ImageFolder format)")
    all_parser.add_argument("--val-dir", required=True, help="Path to real validation dataset (ImageFolder format)")
    all_parser.add_argument("--nclass", type=int, default=10, help="Number of classes")
    all_parser.add_argument("--ipc", type=int, default=10, help="Images per class")
    all_parser.add_argument("--size", type=int, default=224, help="Image resolution")
    all_parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    all_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    all_parser.add_argument("--repeat", type=int, default=3, help="Number of runs per architecture")
    all_parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    all_parser.add_argument("--save-dir", default=None, help="Directory to save results")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        from cspd_eval.train import run_evaluation

        result = run_evaluation(
            distilled_dir=args.distilled_dir,
            val_dir=args.val_dir,
            arch=args.arch,
            nclass=args.nclass,
            ipc=args.ipc,
            size=args.size,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            repeat=args.repeat,
            num_workers=args.num_workers,
            save_dir=args.save_dir,
            epochs=args.epochs,
        )
        print(json.dumps({
            "arch": result["arch"],
            "mean_best_acc1": result["mean_best_acc1"],
            "std_best_acc1": result["std_best_acc1"],
            "mean_best_acc5": result["mean_best_acc5"],
        }, indent=2))

    elif args.command == "run-all":
        from cspd_eval.train import run_evaluation

        all_results = {}
        for arch in ["convnet", "resnet18", "resnet_ap"]:
            print(f"\n{'='*60}")
            print(f"  Architecture: {arch}")
            print(f"{'='*60}")
            result = run_evaluation(
                distilled_dir=args.distilled_dir,
                val_dir=args.val_dir,
                arch=arch,
                nclass=args.nclass,
                ipc=args.ipc,
                size=args.size,
                batch_size=args.batch_size,
                seed=args.seed,
                repeat=args.repeat,
                num_workers=args.num_workers,
                save_dir=args.save_dir,
            )
            all_results[arch] = {
                "mean_best_acc1": result["mean_best_acc1"],
                "std_best_acc1": result["std_best_acc1"],
            }

        print(f"\n{'='*60}")
        print("  Summary (all architectures)")
        print(f"{'='*60}")
        for arch, r in all_results.items():
            print(f"  {arch:12s}: {r['mean_best_acc1']:.1f} +/- {r['std_best_acc1']:.1f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
