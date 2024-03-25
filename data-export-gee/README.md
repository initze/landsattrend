GEE DATA EXPORT - SETUP


1. In order to run this, you will need to create a project that uses google earth engine. The project key currently being used is uiuc-ncsa-permafrost-44d44c10c9c7.json . If you use a json with a different name, you will need to change the code.
2. Additionally, the bucket names may need to change based on your account. These values will later be moved to a config file.

GEE DATA EXPORT - EXPORT TO CLOUD

1. Here is a sample command to export to cloud 
`   export_to_cloud.py --startyear=2010 --endyear=2020 --process_site=ALL `
2. For exporting to cloud, the sites are TEST, ALASKA, CANADA, EURASIA1, EURASIA2, EURASIA3.
3. ALL can also be used and that will export all the sites.
4. Once these jobs are started, it can take up to 48 hours before all the files are in cloud.
5. As the script runs, a list of filenames will appear. Some of those filenames will not actually get exported if they are all water or have no data.

GEE DATA EXPORT - DOWNLOADING DATA

