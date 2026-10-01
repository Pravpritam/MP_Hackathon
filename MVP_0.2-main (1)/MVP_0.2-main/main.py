import os
from video_pipeline import run_pipeline

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_FOLDER   = os.path.join(BASE_DIR, "test")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL         = os.path.join(BASE_DIR, "best.pt")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")

BUFFER_SIZE     = 15
PIXEL_THRESHOLD = 10.0


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_files   = os.listdir(TEST_FOLDER)
    video_files = [f for f in sorted(all_files) if f.lower().endswith(VIDEO_EXTENSIONS)]

    if not video_files:
        print(f"[ERROR] No video files found in '{TEST_FOLDER}/'")
        print(f"        Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        return

    print(f"[INFO] Found {len(video_files)} video(s) in '{TEST_FOLDER}/':")
    for f in video_files:
        print(f"         - {f}")
    print()

    for idx, filename in enumerate(video_files, start=1):
        input_path  = os.path.join(TEST_FOLDER, filename)
        name, ext   = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"{name}_annotated.mp4")

        print(f"[{idx}/{len(video_files)}] Processing: {filename}")
        print(f"            Output  : {output_path}")
        print("-" * 55)

        run_pipeline(
            video_path      = input_path,
            model_path      = MODEL,
            output_path     = output_path,
            buffer_size     = BUFFER_SIZE,
            pixel_threshold = PIXEL_THRESHOLD,
        )
        print()

    print("=" * 55)
    print(f"[DONE] All {len(video_files)} video(s) processed.")
    print(f"       Annotated files saved in '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    main()