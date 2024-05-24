import click
import os 
import numpy as np

@click.command()
@click.option('--input_path', help='Path to orig feature folder where _rgb and _flow.npy has to be combined ')
@click.option('--output_path', help='Path to output feature folder')
def run(input_path, output_path): 
    files = os.listdir(input_path)
    files.sort()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i, f in enumerate(files):
        if i % 2 == 0:
            continue
        a = np.load(os.path.join(input_path, f))
        b = np.load(os.path.join(input_path, files[i-1]))
        
        print(a.shape, b.shape)

        c = np.concatenate((a, b), axis=1)
        out_path = os.path.join(output_path, f.split("_")[0] + "_" + f.split("_")[1] + os.path.splitext(f)[1])
        print("Writing combined features to output path: ", out_path)
        np.save(out_path, c)
        print(c.shape)


if __name__ == '__main__':
    run()
    #run("i3d_step1_stack32_n50_full/", "full")
