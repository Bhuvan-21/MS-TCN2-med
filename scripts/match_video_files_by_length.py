import os
import cv2
import click
from os.path import join
from tqdm import tqdm
import hashlib


@click.command()
@click.option('--videos1', help='Path to folder with videos', type=click.Path(exists=True))
@click.option('--videos2', help='Second path to folder with videos', type=click.Path())
def match_video_files_by_length(videos1, videos2):
    files1 = os.listdir(videos1)
    files2 = os.listdir(videos2)
    
    vid_lengths1 = {}
    vid_lengths2 = {}
    
    for file in tqdm(files1):
        vid_lengths1[file] = create_md5_hash(join(videos1, file))
    for file in tqdm(files2):
        vid_lengths2[file] = create_md5_hash(join(videos2, file))
        
    matching_pairs = []

    for file1, length1 in vid_lengths1.items():
        for file2, length2 in vid_lengths2.items():
            if length1 == length2:
                matching_pairs.append((file1, file2, length1))
    
    print(f'Found {len(matching_pairs)} matching pairs')
    print('Matching pairs:', '\n'.join([str(vid1 + ', ' + vid2 + ': ' + str(length)) for vid1, vid2, length in matching_pairs if vid1 != vid2]))
    
    unique_pairs = set([a for a,b,c in matching_pairs])
    count_unique_pairs = len(unique_pairs)
    print(f'Number of unique pairs: {count_unique_pairs}')
    

def get_video_length(file):
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration, frame_count


def create_md5_hash(file):
    with open(file, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
    return md5_hash


if __name__ == '__main__':
    match_video_files_by_length()