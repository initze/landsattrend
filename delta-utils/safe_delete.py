import os
import sys
current_site_name = sys.argv[1]

path_to_projects = '/projects/bbou/toddn/landsattrend'
path_to_scratch = '/scratch/bbou/toddn/landsat-delta/landsattrend'

def check_data(site_name):
    scratch_data_path = os.path.join(path_to_scratch, 'data', site_name)
    scratch_data_files = os.listdir(scratch_data_path)
    projects_data_path = os.path.join(path_to_projects, 'data', site_name)
    projects_data_files = os.listdir(projects_data_path)

    not_in_scratch = []
    not_same_size = []

    for f in projects_data_files:
        if f not in scratch_data_files:
            not_in_scratch.append(f)
    return not_in_scratch


def check_process(site_name):
    scratch_process_path = os.path.join(path_to_scratch, 'process', site_name)
    if os.path.exists(scratch_process_path):
        scratch_process_files = os.listdir(scratch_process_path)
    else:
        scratch_process_pa
    projects_process_path = os.path.join(path_to_projects, 'process', site_name)
    projects_process_files = os.listdir(projects_process_path)

    not_in_scratch = []

    for f in projects_process_files:
        if f not in scratch_process_files:
            not_in_scratch.append(f)
    return not_in_scratch

if __name__ == "__main__":
    data_files_not_in_scratch = check_data(site_name=current_site_name)
    process_files_not_in_scratch = check_process(site_name=current_site_name)