import subprocess
import sys
import subprocess, os
import time 
from config import *
import random

bucket_name = sys.argv[1]

my_env = os.environ.copy()
my_env['TOKENIZERS_PARALLELISM']='true'

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


masked = list(client.list_blobs(bucket_name, prefix='masked_data/data_'))
masked = [m.name for m in masked]


blobs = []
size = 0
for blob in client.list_blobs(bucket_name, prefix='public_model/corpus/'):
    if(blob.name.endswith('.txt')):
        blobs.append(blob)

indices = [i for i in range(1086,1099)]
data_filter = [195, 193, 159, 141, 200, 204, 218, 225, 226, 457, 458, 465, 511, 517, 665, 768, 769, 800, 948, 964]
data_filter.extend(indices)
blobs_cleaned = [j for i, j in enumerate(blobs) if i not in data_filter]



for blob in blobs_cleaned:
    fn = blob.name
    _fn = slugify(fn)
    
    try:
        upload_data_to_gcs("0", "masked_data_working/%s"%_fn, if_generation_match=0) 
        # ensure just one worker do the job at the same time.
    except:
        continue
    
    if ("masked_data/data_original_%s"%_fn in masked and 
        "masked_data/data_masked_%s"%_fn in masked and 
        "masked_data/data_labels_%s"%_fn in masked
       ):
        continue
        
    t0 = time.time()
    process = subprocess.Popen((sys.executable+' convert_to_bytes.py %s %s'%(bucket_name, fn)).split(' '), env=my_env, stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, b''):
        print(line.decode().strip())
    t1 = time.time()
    
    MB = blob.size / 1024 / 1024
    print(f"** finish processing file {blob.name} in %.4f, speed: %.4f MB / s"%((t1-t0), MB / (t1-t0)))
    print()