import re
import os
import click
import shutil
import subprocess
from tqdm.auto import tqdm

def secs_to_hms(seconds):
    hours = 0 #int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_gt_file(gt_file, fps=30, name_bg="background", name_inf="Fundus_Visibile"):
    with open(gt_file, 'r') as f:
        lines = f.read().split()
    
    if lines[-1] == "":
        lines = lines[:-1]
    
    informative_phases = []
    start_frame = None
    for i, classification in enumerate(lines):
        if classification == name_inf and start_frame is None:
            start_frame = i # Start of an informative phase
        elif classification == name_bg and start_frame is not None:
            end_frame = i - 1 # End of an informative phase
            start_time = secs_to_hms(start_frame / fps)
            end_time = secs_to_hms(end_frame / fps)
            if end_frame - start_frame <= fps:
                end_time = secs_to_hms(end_frame / fps + 1)
            informative_phases.append((start_time, end_time))
            start_frame = None

    if start_frame is not None: # Handle case where the last frames are informative
        start_time = secs_to_hms(start_frame / fps - 1)
        end_time = secs_to_hms(len(lines) / fps)
        informative_phases.append((start_time, end_time))
        
    return informative_phases
    

def get_corresponding_video(gt_file, videos):
    video_id = None
    id_idx = gt_file.find("IMG_")
    if id_idx == -1:
        pattern = r'R\d{3}[RL]\d?' 
        match = re.search(pattern, gt_file)
        if match: video_id = match.group(0)
    else:
        video_id = gt_file[id_idx+4:id_idx + 8]

    assert video_id is not None, f"Could not find video ID in gt_file: {gt_file}" 
    
    for video in videos:
        if video_id in video:
            return video
    return None

def create_merge_file(clip_files, temp_folder):
    merge_file = os.path.join(temp_folder, "file_list.txt")
    with open(merge_file, 'w') as f:
        for clip in clip_files:
            f.write(f"file '{os.path.abspath(clip)}'\n")
    return merge_file
            

@click.command()
@click.option('--input_path', '-i', help='Path to the input folder (holding uncut videos)', required=True)
@click.option('--output_path', '-o', help='Path to the output folde for merged extracted clips', required=True)
@click.option('--phases_path', '-p', help='Path to the folder holding phase information for input vidoes (which areas are informative)', required=True)
def run(input_path, output_path, phases_path):
    videos = os.listdir(input_path)
    gt_files = os.listdir(phases_path)
    os.makedirs(output_path, exist_ok=True)
    
    temp_folder = "/tmp/temp_clips"
    for gt in tqdm(gt_files, total=len(gt_files)):
        if os.path.exists(temp_folder): shutil.rmtree(temp_folder) # Clean-up
        os.makedirs(temp_folder, exist_ok=False)
        informative_phases = read_gt_file(os.path.join(phases_path, gt))
        video_path = get_corresponding_video(gt, videos)
        # assert video_path is not None, f"Could not find corresponding video for gt file: {gt}"
        if video_path is None:
            print(f"Video match for {gt} not found. Skipping...")
            continue
        elif os.path.exists(os.path.join(output_path, video_path)):
            print(f"Clips were already extracted from {video_path}. Continuing with next file...")
            continue
        
        clip_files = []
        
        for i, (start, end) in enumerate(informative_phases):
            output_clip = os.path.join(temp_folder, f"clip_{i + 1}.mp4")
            print(f"Extracting clip {i + 1}: {start} to {end} -> {output_clip}")

            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", os.path.join(input_path, video_path), "-ss", start, "-to", end, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", output_clip, "-y"], check=False)
            clip_files.append(output_clip)
        
        merge_file = create_merge_file(clip_files, temp_folder) # Create a file list for merging
        # print(f"Merging clips into {output_path}")
        output_file = os.path.join(output_path, video_path)
        
        if len(clip_files) > 0:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", merge_file, "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an", output_file, "-y"], check=True)
        
        print(f"Final merged video saved to {output_file}")


if __name__ == '__main__':
    run()
