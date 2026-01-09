import cv2
import torch
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# set parameters
TARGET_SIZE = (224, 224)
NUM_FRAMES = 8

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def uniform_sample_frames(total_frames, num_frames):
    if total_frames <= num_frames:
        return list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    interval = total_frames / num_frames
    return [int(i * interval) for i in range(num_frames)]

def extract_and_save(video_dir, save_dir, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_dir)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video has no frames - {video_dir}")
        cap.release()
        return
    indices = uniform_sample_frames(total_frames, num_frames)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb)
            frames.append(tensor)
    cap.release()
    if len(frames) < num_frames:
        print(f"Frame completion: {video_dir} (original {len(frames)}帧 -> to {num_frames}帧)")
        last_frame = frames[-1] if frames else torch.zeros(3, *TARGET_SIZE)
        frames.extend([last_frame] * (num_frames - len(frames)))
    if len(frames) != num_frames:
        print(f"Error: Still not enough frames {len(frames)} != {num_frames} - {video_dir}")
        return
    frames_tensor = torch.stack(frames)
    torch.save(frames_tensor, save_dir)

def process_dataset(video_dir, save_dir, suffix = ".mp4"):
    if not os.path.exists(video_dir):
        print(f"the path of {video_dir} does not exist")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fname in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, fname)
        save_path_temp = os.path.join(save_dir, fname)
        for vname in os.listdir(video_path):
            video_name_path = os.path.join(video_path, vname)
            save_name = os.path.splitext(vname)[0] + '.pt'
            save_path = os.path.join(save_path_temp, save_name)
            extract_and_save(video_name_path, save_path)


if __name__ == '__main__':
    dataset_name = "SIMS"# MOSI, MOSEI, SIMS
    input_video_dir = f"/{dataset_name}/Raw/"
    output_tensor_dir = f"{dataset_name}/wav"
    process_dataset(input_video_dir, output_tensor_dir)




