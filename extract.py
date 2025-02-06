import os
import cv2
import argparse
import ffmpeg

def extract_frames(video_path, output_dir, frame_rate=30, width=480, height=270):
    """
    Extracts frames from a video using ffmpeg and saves them as grayscale images in the specified directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        frame_rate (int): Frames per second to extract.
        width (int): Resized width of the frames.
        height (int): Resized height of the frames.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Extracting frames from {video_path}...")

    # Use ffmpeg to extract frames
    ffmpeg.input(video_path).output(
        os.path.join(output_dir, 'frame_%06d.png'),
        vf=f'scale={width}:{height},format=gray',  # Resize & Convert to Grayscale
        r=frame_rate  # Extract at specified frame rate
    ).run(overwrite_output=True)

    print(f"Frames extracted and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them as grayscale images.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted frames.")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frames per second to extract (default: 30).")
    
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.frame_rate)
