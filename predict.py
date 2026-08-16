import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def run_inference(source, model_path="best.pt", conf=0.25, save=True, show=False, project="output", name="results", exist_ok=True):
    """
    Run object detection inference using the trained YOLO model.
    """
    print(f"Loading model: {model_path} ...")
    model = YOLO(model_path)
    
    print(f"Running inference on source: {source} (Confidence threshold: {conf})")
    print(f"Saving output to: {project}/{name} ...")
    
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=show,
        project=project,
        name=name,
        exist_ok=exist_ok,
        stream=True  # stream for efficient memory handling on videos
    )
    
    detection_count = 0
    frames_processed = 0
    save_dir = None
    
    for r in results:
        frames_processed += 1
        if save_dir is None and hasattr(r, "save_dir"):
            save_dir = r.save_dir
            
        boxes = r.boxes
        if len(boxes) > 0:
            detection_count += len(boxes)
            detected_classes = [model.names[int(b.cls[0].item())] for b in boxes]
            if frames_processed % 30 == 0 or frames_processed == 1:
                print(f"  [Frame {frames_processed}] Detected: {', '.join(detected_classes)}")

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total Frames Processed: {frames_processed}")
    print(f"Total Sign Detections: {detection_count}")
    if save_dir:
        print(f"[+] Output saved to folder: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Inference with YOLO")
    parser.add_argument("--source", type=str, default="test.jpg", help="Path to image, video, folder, or '0' for webcam")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to weights file (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--project", type=str, default="output", help="Output directory folder name (default: output)")
    parser.add_argument("--name", type=str, default="results", help="Subdirectory name inside project (default: results)")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--no-save", action="store_true", help="Do not save output images/videos")
    
    args = parser.parse_args()
    run_inference(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=not args.no_save,
        show=args.show
    )

