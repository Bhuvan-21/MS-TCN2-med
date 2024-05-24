import click
import os

@click.command()
@click.option('--input_path','-i', help='Path to the input file')
@click.option('--input_file','-f', help='Input file name with list of files')
@click.option('--output_file','-o', help='Output file name')
def get_ground_truth_frequency(input_path, input_file, output_file, fps=15):
    click.echo(f'Calculating frequencies for gt in {input_path}')
    click.echo(f'Calculating frequencies for gt files in {input_file}')
    
    
    if input_path and not input_file:
        files = os.listdir(input_path)
    elif input_file and input_path:
        files = open(input_file).read().split('\n')
        if files[-1] == '': files = files[:-1]
    else:
        click.echo('Please provide either input path or input file')
        return
    
    class_count = {}
    for file in files:
        lines = open(os.path.join(input_path, file)).readlines()
        
        for line in lines[1:]:
            line = line.strip('\n')
            if class_count.get(line) is None:
                class_count[line] = 1
            class_count[line] += 1
    
    class_count = {k: v / float(fps) for k,v in class_count.items()}
    frequencies = {k: round(v / sum(class_count.values()), 4) for k,v in sorted(class_count.items(), key=lambda item: item[1])}
    #print('Detected class clount ', class_count)   
    print('Found classes: ', ', '.join(class_count.keys()))
    print('Detected class fequencies ', frequencies)
    
    if output_file:
        with open(output_file, 'w') as f:
            for k, v in frequencies.items():
                f.write(f'{k} {v}\n')
        
    

if __name__ == '__main__':
    get_ground_truth_frequency()
