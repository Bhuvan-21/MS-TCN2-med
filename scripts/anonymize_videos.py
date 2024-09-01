import click
import os
import cv2
from tqdm.auto import tqdm


@click.command()
@click.option('--input_dir', '-i', required=True, help='Input directory containing videos')
@click.option('--output_dir', '-o', required=True, help='Output directory to save anonymized videos')
def anonymize_videos(input_dir, output_dir):
    videos = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for video in tqdm(videos):
        video_path = os.path.join(input_dir, video)
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        box_height = height // 15
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(os.path.join(output_dir, video), fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply Gaussian blur to the first 100 rows
            frame[:box_height, 220:] = cv2.boxFilter(frame[:box_height, 220:], -1, (15, 15)) #cv2.blur(frame[:box_height, :], (20, 20), 0)
            out.write(frame) # Write the frame to the output video
        
        cap.release()
        out.release()
        # break # stop after processing one video
    

if __name__ == '__main__':
    anonymize_videos()
    
