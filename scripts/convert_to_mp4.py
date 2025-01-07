import os
import subprocess
import click

@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
def convert_videos(input_dir, output_dir):
    """
    Converts all .MOV files in INPUT_DIR (and subdirectories) to .mp4 format 
    and saves them in OUTPUT_DIR, preserving the folder structure.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Walk through the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mov'):  # Check for .MOV files
                input_path = os.path.join(root, file)

                # Construct the relative path to maintain folder structure
                rel_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, rel_path)
                os.makedirs(output_folder, exist_ok=True)

                # Define the output file path
                output_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.mp4')

                # Call ffmpeg to convert the file
                print(f"Converting: {input_path} -> {output_path}")
                subprocess.run(['ffmpeg', '-i', input_path, '-c', 'copy', output_path, '-y'], check=True)

if __name__ == '__main__':
    convert_videos()

