This section is for uploading files to and from delta.

1. cloud_utils.py
   1. This will download files from our google cloud console into Delta. 
   2. command: ```` python cloud_util.py 32606````
   3. Autnetication is handled by the file that should be in project-key. That json file is the credentials for the cloud account.
2. upload_input.py
   1. This uploads the INPUT for a site.
   2. command: ```` python upload_input.py {clowder_url} {clowder-key} {current_site}````
3. upload_output.py
4. 1. This uploads the INPUT for a site.
   2. command: ```` python upload_ouput.py {clowder_url} {clowder-key} {current_site}````
5. show_process_num_files.py
   1. This will list the files that are already in clowder, and whether they are the right size.
   2. command ```python show_process_num_files.py {site_name} {clowder_url} {key} ```