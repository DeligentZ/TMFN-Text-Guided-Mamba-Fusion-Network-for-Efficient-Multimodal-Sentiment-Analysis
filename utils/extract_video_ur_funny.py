
import cv2
import numpy as np
import os
from tqdm import tqdm


TARGET_SIZE = (224, 224)  # ViT input size
NUM_FRAMES = 8

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



def uniform_sample_frames(total_frames, num_frames):
    if total_frames <= num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        return indices
    interval = total_frames / num_frames
    return [int(i * interval) for i in range(num_frames)]


def preprocess_frame(frame):
    # Resize
    frame = cv2.resize(frame, TARGET_SIZE)

    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # uint8 [0, 255] -> float32 [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # Normalize: (x - mean) / std
    frame = (frame - MEAN) / STD

    # HWC -> CHW: (224, 224, 3) -> (3, 224, 224)
    frame = frame.transpose(2, 0, 1)

    return frame


def extract_single_video(video_path, num_frames=NUM_FRAMES):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None,

    indices = uniform_sample_frames(total_frames, num_frames)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            processed = preprocess_frame(frame)
            frames.append(processed)

    cap.release()

    # 检查是否成功读取
    if len(frames) == 0:
        return None,

    # 补帧 (如果不足)
    if len(frames) < num_frames:
        last_frame = frames[-1]
        frames.extend([last_frame] * (num_frames - len(frames)))

    # Stack: [num_frames, 3, 224, 224]
    frames_array = np.stack(frames, axis=0).astype(np.float32)

    return frames_array, None


def extract_video_urfunny(input_dir, output_dir, num_frames=NUM_FRAMES):

    if not os.path.exists(input_dir):
        print(f"error: the input path is not exist - {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"creat output path: {output_dir}")
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    print(f"find {len(video_files)} visula files")
    print(f"sample frame: {num_frames}")
    print(f"output size: [{num_frames}, 3, {TARGET_SIZE[0]}, {TARGET_SIZE[1]}]")
    print(f"output format: .npy (NumPy array)")

    success_count = 0
    skip_count = 0
    error_count = 0
    error_files = []

    for video_file in tqdm(video_files, desc="extract video frame"):
        input_path = os.path.join(input_dir, video_file)

        output_filename = video_file.replace('.mp4', '.npy')
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            skip_count += 1
            continue

        try:
            frames_array, error_msg = extract_single_video(input_path, num_frames)

            if frames_array is None:
                error_files.append((video_file, error_msg))
                error_count += 1
                continue

            np.save(output_path, frames_array)
            success_count += 1

        except Exception as e:
            error_files.append((video_file, str(e)))
            error_count += 1

    print("\n" + "=" * 50)
    print("extract completed")
    print(f"success: {success_count}")
    print(f"skip(exist): {skip_count}")
    print(f"fail: {error_count}")
    print(f"output format: .npy, shape=[{num_frames}, 3, {TARGET_SIZE[0]}, {TARGET_SIZE[1]}]")
    print("=" * 50)

    if error_files:
        print("\nlist of fail file (前10个):")
        for fname, reason in error_files[:10]:
            print(f"  - {fname}: {reason}")
        if len(error_files) > 10:
            print(f"  ...  and {len(error_files) - 10} files")

if __name__ == "__main__":
    INPUT_VIDEO_DIR = ""
    OUTPUT_VIDEO_DIR = ""

    if not INPUT_VIDEO_DIR or not OUTPUT_VIDEO_DIR:
        print("Pls setting INPUT_VIDEO_DIR and OUTPUT_VIDEO_DIR path!")
        print("example:")
        print('  INPUT_VIDEO_DIR = "/data/datasets/urfunny2_videos"')
        print('  OUTPUT_VIDEO_DIR = "/data/datasets/URFUNNY/video"')
    else:
        extract_video_urfunny(INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR)
