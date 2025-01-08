import os
import click
import subprocess
from pathlib import Path
from tqdm import tqdm


@click.command()
@click.option('--source_folder', '-i', type=click.Path(exists=True))
@click.option('--output_folder', '-o', default='temp/', help='Output folder for rescaled videos')
@click.option('--center_crop', '-c', is_flag=True, help='Center crop the video before rescaling')
def rescale_videos(source_folder, output_folder, center_crop):
    if os.path.exists(output_folder):
        click.echo('Output path already exists...')
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all video files in the source folder
    video_files = [f for f in os.listdir(source_folder) if f.endswith(('.mp4', '.avi', '.mkv', '.MOV'))]

    # Iterate over each video file
    for video_file in tqdm(video_files, total=len(video_files)):
        # Construct the input and output file paths
        input_file = Path(os.path.join(source_folder, video_file)).resolve()
        output_file = Path(os.path.join(output_folder, video_file)).resolve()
        print(input_file)

        if os.path.exists(output_file):
            continue

        # Run the ffmpeg command to scale the video
        if not center_crop:
            ffmpeg_command = f'ffmpeg -stats -v repeat+level+warning -i "{input_file}" -vf scale=iw/2:ih/2:flags=lanczos -c:v libx264 -preset medium -crf 18 "{output_file}"'
            os.system(ffmpeg_command)
        else:
            command = ["ffmpeg", "-stats", "-v", "repeat+level+warning", "-i", input_file, "-vf",
                       "crop='min(in_w, in_h)':'min(in_w, in_h)':'(in_w-min(in_w, in_h))/2':'(in_h-min(in_w, in_h))/2', scale=iw/2:ih/2:flags=lanczos",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "18", output_file]
            # Run the FFmpeg command
            subprocess.run(command, check=True)

    click.echo("Videos rescaled successfully!")


if __name__ == '__main__':
    rescale_videos()
