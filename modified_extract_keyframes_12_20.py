import os, subprocess

file_type = '.mp4'
file_types = ['.mp4', '.MXF', '.mxf']


def run_command(command, logfile=None, print_output=False, return_output=False):
    if logfile != None:
        command += ' |& tee ' + logfile
    output = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        executable='/bin/bash'
    ).stdout.read()
    if print_output:
        print(output)
    if return_output:
        return str(output)


# list to store the names of the files
mfiles_to_extract = []
source_roots = "/dvmm-filer2/projects/Hearst/data_new"
# count how many .mp4, .mxf, .MXF are in the source root - don't count duplicates
file_counter = 0
for root, dirs, files in os.walk(source_roots):
    for file in files:
        # for ft in file_types:
        for f_type in file_types:
            if file.endswith(f_type) and file not in mfiles_to_extract:
                file_counter += 1
                file_name_w_path = os.path.join(root, file)
                print(file_name_w_path)
                mfiles_to_extract.append(file_name_w_path)

print(f'Total # of files that end with .mp4, .MXF, .mxf: {len(mfiles_to_extract)} \n from {mfiles_to_extract}.')

# extract
dest_root = '/dvmm-filer2/projects/Hearst/keyframes/keyframes2_dec_20'

unsuc = []
for i, filename in enumerate(mfiles_to_extract):
    dest_dir = ''

    if os.path.exists(dest_dir):
        print('path exists - SKIPPING')
        continue

    print('extraction about to start in: ')
    print(dest_dir)
    run_command(f'mkdir {dest_dir}')
    out = run_command(
        f"ffmpeg -i {filename} -vf select='eq(n\,1)+gt(scene\,0.2)' -vsync vfr {dest_dir}/frame%05d.png",
        logfile=f'{dest_dir}/log.txt',
        return_output=True
    )
    if 'failed' in out:
        print(f'Processing {filename} failed.----*******-------')
        unsuc.append(filename)

    print(f'{i+1} out of {len(mfiles_to_extract)} videos processed.')

with open(os.path.join(dest_root, 'unsuc.txt'), 'w') as fout:
    for item in unsuc:
        fout.write(unsuc)
        fout.write('\n')
