# cython: profile=False, embedsignature=True, boundscheck=False, wraparound=True, nonecheck=False, cdivision=True, language_level=2, language=c++
import os
import time

from google.cloud import storage
client = storage.Client()
def upload_data_to_gcs(data, target_key):
    bucket = client.bucket(bucket_name)
    bucket.blob(target_key).upload_from_string(data)
    return bucket.blob(target_key).public_url


import numpy as np
import random

import jieba
import logging
import requests
logging.getLogger("jieba").setLevel(logging.WARNING)

r = requests.get('https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big')
with open('dict.txt.big' , 'w') as f: 
    f.write(r.text)
jieba.load_userdict('dict.txt.big')
jieba.add_word('<nl>')
c = requests.get('https://gist.githubusercontent.com/ecchochan/23613d281f5bbeab97a6a5b9318e902a/raw/91f6496dd1c71b7398dcfd4915f1113c40f0ac22/jieba_hk.py')





def iterator_gen(generator, handler=None, parallel = False, chunksize =128):
    try:
        import gc
        import multiprocessing as multiprocessing
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cpu_count)

        if handler is not None:
            for e in (pool.imap_unordered(handler,generator, chunksize=chunksize ) if parallel else iter(handler(e) for e in generator)):
                if e:
                    if isinstance(e, list):
                        for e in e:
                            yield e
                    else:
                        yield e
                    
        else:
            for e in generator:
                if e:
                    if isinstance(e, list):
                        for e in e:
                            yield e
                    else:
                        yield e
    finally:
        if parallel:
            try:
                pool.terminate()
            except:
                pass
            gc.collect()
            

import math
import collections
def weigthed_shuffle(items, weights, mask_no):
    order = sorted(range(len(items)), key=lambda i: -random.random() ** (1.0 / weights[i]))[:mask_no]
    return [items[i] for i in order]


def split_doc(doc, length_limit = 10000):
    bucket_lines = []
    buc = []
    accum = 0
    for line in doc.split('\n'):
        line = line.strip()
        buc.append(line)
        length = len(line)
        if length > length_limit:
            continue
        accum += length
        if accum > length_limit:
            accum = 0
            if buc:
                bucket_lines.append('\n'.join(buc))

    if buc:
        bucket_lines.append('\n'.join(buc))

    return bucket_lines



# Prepare something for worker to do
def generator():
    index = 0
    for item in encoded:
        yield (item, index)
        index += 1

# Actual Work
def worker(item):
    size = 0
    item, index = item
    
    char2tokens = {}
    
    for tok_i, (a, b) in enumerate(item.offsets):
        if tok_i == 0:
            continue
        for i in range(a, b + 1):
            if a == 0 and b == 0:
                continue
            char2tokens[i] = tok_i

    encoding_ids = item.ids

    local_tok_to_freqs = collections.Counter(encoding_ids)


    pickables = []
    weights = []
    seen = set()
    for j, a, b in jieba.tokenize(data[index]):
        if not j.strip():
            continue
        toks = {char2tokens[i] for i in range(a, b) if i in char2tokens and char2tokens[i] not in seen}
        for e in toks:
            seen.add(e)
        if not toks:
            continue
        freq_sum = 0
        tok_count = 0
        tok_is = []

        local_f = 0

        for tok_i in toks:
            tok = encoding_ids[tok_i]
            if tok < 6:
                continue
            freq = tot_count[tok]
            local_f += local_tok_to_freqs[tok]
            freq_sum += freq
            tok_count += 1
            tok_is.append(tok_i)
            
        if tok_count == 0:
            # cross document, abort
            continue
        assert tok_count > 0

        freq_sum = (freq_sum**0.5) * local_f

        weight = 1 / (1 + freq_sum / tok_count) * 10

        pickables.append(tok_is) # tok_i  s
        weights.append(weight )


    ids, ret_labels, ret_masked_ids = [], [], []
    ids = np.array(encoding_ids, dtype = np.int16)
    labels = np.full_like(encoding_ids, -100, dtype=np.int16)
    masked_ids = ids.copy()
    masked_words = 0


    mask_no = int(round((len(encoding_ids) - (np.array(encoding_ids)<=5).sum()).item()*0.15))
    sample = weigthed_shuffle(pickables, weights, mask_no)

    for toks in sample:
        toks.sort()
        start_index = toks[0]
        end_index = toks[-1]

        if start_index is None or end_index is None:
            continue
        else:
            if start_index == 0:
                start_index += 1
            masked_words += end_index - start_index + 1
            labels[start_index:end_index+1] = ids[start_index:end_index+1]
            p = random.random()
            if p <=0.1:
                masked_ids[start_index:end_index+1] = np.random.randint(6,50000,size = end_index - start_index + 1)
            elif p <=0.9:
                masked_ids[start_index:end_index+1] = 4
            else:
                masked_ids[start_index:end_index+1] = ids[start_index:end_index+1]
        if masked_words > mask_no:
            break


    if(np.count_nonzero(ids) <= 15) or np.count_nonzero(np.array(ids) < 7) > 5:
        return
        

    if len(ids) == 0:
        return None

    chunk_size = 128
    # take away the [CLS] & [SEP] & [PAD]
    ids_flat = ids
    labels_flat = labels
    masked_ids_flat = masked_ids

    # Append 0 at the end
    ids_flat = np.append(ids_flat, 0)
    masked_ids_flat = np.append(masked_ids_flat, 0)
    labels_flat = np.append(labels_flat, 0)

    # Determine the number of chunks
    chunks_size = -(-ids_flat.size // chunk_size) * chunk_size

    # Pad the chunks
    ids_flat_chunks = np.zeros(chunks_size, dtype=np.int16)
    ids_flat_chunks[:ids_flat.size] = ids_flat
    masked_ids_flat_chunks = np.zeros(chunks_size, dtype=np.int16)
    masked_ids_flat_chunks[:masked_ids_flat.size] = masked_ids_flat
    labels_flat_chunks = np.zeros(chunks_size, dtype=np.int16)
    labels_flat_chunks[:labels_flat.size] = labels_flat

    if np.count_nonzero(ids_flat_chunks[-chunk_size:]) < 15:
        if ids_flat_chunks.size <= chunk_size:
            return None

        ids_flat_chunks = ids_flat_chunks[:-chunk_size]
        ids_flat_chunks[-1] = 0
        masked_ids_flat_chunks = masked_ids_flat_chunks[:-chunk_size]
        masked_ids_flat_chunks[-1] = 0
        labels_flat_chunks = labels_flat_chunks[:-chunk_size]
        labels_flat_chunks[-1] = 0


    return ids_flat_chunks.tobytes(), labels_flat_chunks.tobytes(), masked_ids_flat_chunks.tobytes()

from joblib import Parallel, delayed
from time import sleep
def fetch(blob):
    return blob.download_as_string()


import sys
bucket_name = sys.argv[1]
fn = sys.argv[2]
blob = list(client.list_blobs(bucket_name, prefix=fn))[0]


data = blob.download_as_string()


if not data:
    import sys
    print('exiting')
    sys.exit()


t0 = time.time()
flat_data = []
for doc in data.decode().split('\n\n'):
    for sub_doc in split_doc(doc):
        if not sub_doc.strip():
            continue
        flat_data.append(sub_doc)

t1 = time.time()

print('split, %.4f'%(t1-t0))
data = flat_data
print(f"start tokenizing file {blob.name} {len(flat_data)}")
sleep(0.4)
t0 = time.time()


def get_encoded():
    import tokenizers
    from transformers import BertTokenizer
    from cantokenizer import CanTokenizer
    seq_length = 512
    tokenizer = CanTokenizer(vocab_file = 'cantokenizer-vocab.txt')
    encoded = tokenizer.encode_batch(data)
    return encoded
encoded = get_encoded()

t1 = time.time()
print(f"finish tokenizing file {blob.name}, %.4f"%(t1-t0))


t0 = time.time()


count = {}
for item in encoded:
    for a in item.ids:
        if not a or a == 2:
            continue
        if a not in count:
            count[a] = 0
        count[a] += 1

t1 = time.time()
print('count, time: %.4f'%(t1-t0))

tot_count = count

g = generator()
print(f"start masking file {blob.name}")
j = 0


buffer_ids = []
buffer_labels = []
buffer_masked_ids = []
t0 = time.time()
for ret in iterator_gen(g, worker, parallel=True):
    if ret is None:
        continue
    ids, labels, masked_ids = ret
    buffer_ids.append(ids)
    buffer_masked_ids.append(masked_ids)
    buffer_labels.append(labels)

t1 = time.time()
print(f"finish masking file {blob.name} time: %.4f"%(t1-t0))

import unicodedata, re
def slugify(value):
    value = unicodedata.normalize('NFKD', value.split('/')[-1].split('.')[0])
    value = (re.sub(r'[^\w\s-]', '', value).strip().lower())
    value = (re.sub(r'[-\s]+', '-', value))
    return value


_fn = slugify(fn)


t0 = time.time()
upload_data_to_gcs(b''.join(buffer_ids),"masked_data/data_original_%s"%_fn)
upload_data_to_gcs(b''.join(buffer_labels),"masked_data/data_masked_%s"%_fn)
upload_data_to_gcs(b''.join(buffer_masked_ids),"masked_data/data_labels_%s"%_fn)


t1 = time.time()
print(f"uploaded file {blob.name} time: %.4f"%(t1-t0))



