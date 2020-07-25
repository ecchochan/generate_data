import subprocess
import sys
import subprocess, os
import time 
import random

from google.cloud import storage

client = storage.Client()

def upload_data_to_gcs(data, target_key, **kwargs):
    bucket = client.bucket(bucket_name)
    bucket.blob(target_key).upload_from_string(data, **kwargs)
    return bucket.blob(target_key).public_url

import unicodedata, re
def slugify(value):
    value = unicodedata.normalize('NFKD', value.split('/')[-1].split('.')[0])
    value = (re.sub(r'[^\w\s-]', '', value).strip().lower())
    value = (re.sub(r'[-\s]+', '-', value))
    return value



##########################################
####                                  ####
#### Arguments passed to this scripts ####
####                                  ####
##########################################

bucket_name = sys.argv[1]



#############################################
##                                         ##
##   Get the filenames that are complete   ##
##                                         ##
#############################################

masked = list(client.list_blobs(bucket_name, prefix='masked_data/data_'))
masked = [m.name for m in masked]



###########################################
##                                       ##
##   Get the data that need processing   ##
##                                       ##
###########################################

blobs = []
for blob in client.list_blobs(bucket_name, prefix='public_model/corpus/'):
    if(blob.name.endswith('.txt')):
        blobs.append(blob)


       

###############################
##                           ##
##   My hardcoded cleaning   ##
##                           ##
###############################

indices = [i for i in range(1086,1099)]
data_filter = [195, 193, 159, 141, 200, 204, 218, 225, 226, 457, 458, 465, 511, 517, 665, 768, 769, 800, 948, 964]
data_filter.extend(indices)
blobs = [j for i, j in enumerate(blobs) if i not in data_filter]




#####################################
##                                 ##
##   Set env variables if needed   ##
##                                 ##
#####################################

my_env = os.environ.copy()
my_env['TOKENIZERS_PARALLELISM']='true'




for blob in blobs:
    fn = blob.name
    _fn = slugify(fn)
    
    try:
        #####################################
        ##                                 ##
        ##      Here is the lock logic     ##
        ##   Go next if the job is locked  ##
        ##                                 ##
        #####################################

        upload_data_to_gcs("0", "masked_data_working/%s"%_fn, if_generation_match=0) 
        # ensure just one worker do the job at the same time.

    except:
        continue

    #######################################################
    ##                                                   ##
    ##   Check if the processed data are already there   ##
    ##                                                   ##
    #######################################################

    if ("masked_data/data_original_%s"%_fn in masked and 
        "masked_data/data_masked_%s"%_fn in masked and 
        "masked_data/data_labels_%s"%_fn in masked
       ):
        continue
        



    t0 = time.time()

    ###############################################
    ##                                           ##
    ##   Run the script to do the actual work!   ##
    ##                                           ##
    ###############################################
    process = subprocess.Popen((sys.executable+' convert_to_bytes.py %s %s'%(bucket_name, fn)).split(' '), env=my_env, stdout=subprocess.PIPE)



    for line in iter(process.stdout.readline, b''):
        print(line.decode().strip())
    t1 = time.time()
    
    MB = blob.size / 1024 / 1024
    print(f"** finish processing file {blob.name} in %.4f, speed: %.4f MB / s"%((t1-t0), MB / (t1-t0)))
    print()