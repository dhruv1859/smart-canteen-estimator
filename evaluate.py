"""
evaluate.py  –  Offline precision/recall evaluation script
─────────────────────────────────────────────────────────────────────────────
Usage:
    python evaluate.py --gt data/gt_labels.csv --images data/sample_frames/

Expects gt_labels.csv with columns:
    filename, x1, y1, x2, y2

Prints: Precision, Recall, F1, TP, FP, FN
"""

import argparse, csv, os, cv2
from src.pipeline import CanteenPipeline
from src.detector import _iou


def load_gt(csv_path: str) -> dict:
    """Load ground-truth boxes grouped by image filename."""
    gt: dict[str, list] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            box   = [int(row["x1"]), int(row["y1"]),
                     int(row["x2"]), int(row["y2"])]
            gt.setdefault(fname, []).append(box)
    return gt


def evaluate(gt_path: str, images_dir: str,
             model: str = "yolov8n.pt", conf: float = 0.35,
             iou_thresh: float = 0.5) -> dict:
    pipe = CanteenPipeline(model_path=model, conf=conf, enable_log=False)
    gt   = load_gt(gt_path)

    total_tp = total_fp = total_fn = 0

    for fname, gt_boxes in gt.items():
        img_path = os.path.join(images_dir, fname)
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue
        frame  = cv2.imread(img_path)
        result = pipe.process_frame(frame, show_heatmap=False)
        pred_boxes = [d.box for d in result["detections"]]

        metrics = pipe.detector.evaluate(gt_boxes, pred_boxes, iou_thresh)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        print(f"{fname}: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    summary = dict(precision=precision, recall=recall, f1=f1,
                   tp=total_tp, fp=total_fp, fn=total_fn)
    print("\n── Evaluation Summary ──────────────────────────")
    for k, v in summary.items():
        print(f"  {k:12s}: {v:.4f}" if isinstance(v, float) else f"  {k:12s}: {v}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",     required=True, help="Path to GT CSV file")
    parser.add_argument("--images", required=True, help="Directory of test images")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--conf",   type=float, default=0.35)
    parser.add_argument("--iou",    type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.gt, args.images, args.model, args.conf, args.iou)

