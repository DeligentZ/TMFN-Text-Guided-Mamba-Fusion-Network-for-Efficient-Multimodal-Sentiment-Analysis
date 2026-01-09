import os
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio_urfunny(input_dir, output_dir, sample_rate=16000):
    if not os.path.exists(input_dir):
        print(f"no {input_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} has been created!")
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    print(f"Find {len(video_files)} video files")
    success_count = 0
    skip_count = 0
    error_count = 0
    error_files = []
    for video_file in tqdm(video_files, desc = "extract audio urfunny"):
        input_path = os.path.join(input_dir, video_file)
        output_filename = video_file.replace(".mp4", ".wav")
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            skip_count += 1
            continue
        try:
            video = VideoFileClip(input_path)
            if video.audio is None:
                print(f"warning: no audio track {video_file}")
                error_files.append(video_file)
                error_count += 1
                video.close()
                continue
            video.audio.write_audiofile(
                output_path,
                fps=sample_rate,
                codec='pcm_s16le',
                logger=None
            )
            video.close()
            success_count += 1
        except Exception as e:
            print(f"\n错误: {video_file} - {e}")
            error_files.append((video_file, str(e)))
            error_count += 1

            # 6. 打印统计信息
    print("\n" + "=" * 50)
    print("音频提取完成!")
    print(f"成功: {success_count}")
    print(f"跳过 (已存在): {skip_count}")
    print(f"失败: {error_count}")
    print("=" * 50)

    if error_files:
        print("\n失败文件列表:")
        for fname, reason in error_files[:10]:
            print(f"  - {fname}: {reason}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个文件")

if __name__ == '__main__':
    input_video_dir = ''
    output_video_dir = ''

    if not input_video_dir or not output_video_dir:
        print("pls set the path of input/output video dir")
    else:
        extract_audio_urfunny(input_video_dir, output_video_dir)