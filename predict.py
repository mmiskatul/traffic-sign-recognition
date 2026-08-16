import argparse
import os
import subprocess
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

def convert_to_h264(input_path, output_path):
    """
    Convert video to H.264 / YUV420p for 100% compatibility with LinkedIn, browsers, and social media.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "main",
            "-movflags", "+faststart",
            str(output_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except Exception as e:
        print(f"  [Notice] H.264 conversion skipped: {e}")
        return False

def run_inference(source, model_path="best.pt", conf=0.25, save=True, show=False, project="output", name="results", exist_ok=True):
    """
    Run object detection inference using the trained YOLO model with LinkedIn/Web compatible MP4 video saving.
    """
    print(f"Loading model: {model_path} ...")
    model = YOLO(model_path)
    
    out_dir = Path(project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source)
    is_video_file = source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS
    is_webcam = source == "0" or source.isdigit()

    if is_video_file:
        print(f"Running video inference on: {source} (Confidence threshold: {conf})")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        final_output_video = out_dir / f"{source_path.stem}_detected.mp4"
        raw_temp_video = out_dir / f"temp_{source_path.stem}_raw.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_temp_video), fourcc, fps, (width, height)) if save else None

        frame_idx = 0
        detection_count = 0

        print(f"Processing {total_frames} frames -> Target Output: {final_output_video} (Format: MP4 H.264, {fps:.1f} FPS, {width}x{height})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            results = model.predict(frame, conf=conf, verbose=False)
            r = results[0]
            annotated_frame = r.plot()

            if writer:
                writer.write(annotated_frame)

            boxes = r.boxes
            if len(boxes) > 0:
                detection_count += len(boxes)
                detected_classes = [model.names[int(b.cls[0].item())] for b in boxes]
                if frame_idx % 30 == 0 or frame_idx == 1:
                    print(f"  [Frame {frame_idx}/{total_frames}] Detected: {', '.join(detected_classes)}")

            if show:
                cv2.imshow("Traffic Sign Recognition", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        # Convert to standard H.264 (YUV420p) for LinkedIn / Browser support
        if save and raw_temp_video.exists():
            print("Encoding video to web/LinkedIn-ready H.264 MP4 format...")
            success = convert_to_h264(raw_temp_video, final_output_video)
            if success:
                raw_temp_video.unlink(missing_ok=True)
            else:
                raw_temp_video.rename(final_output_video)

        print("\n" + "=" * 50)
        print("VIDEO PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total Frames Processed: {frame_idx}")
        print(f"Total Sign Detections: {detection_count}")
        if save:
            print(f"[+] Output LinkedIn-Ready MP4 video saved to: {final_output_video}")

    else:
        # Images, directory, or webcam
        print(f"Running inference on source: {source} (Confidence threshold: {conf})")
        print(f"Saving output to: {out_dir} ...")
        
        results = model.predict(
            source=source,
            conf=conf,
            save=save,
            show=show,
            project=project,
            name=name,
            exist_ok=exist_ok,
            stream=True
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
                print(f"  [Item {frames_processed}] Detected: {', '.join(detected_classes)}")

        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total Processed: {frames_processed}")
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



