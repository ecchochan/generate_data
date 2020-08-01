## Distributed Pre-training Data Pre-processing 🐢 + 💰 = 🐢💨💨

### **Purpose**

This is an attempt to distrubte heavy workload across multiple Google VMs through Google Instance Group, utilizing 1000+ CPUs.

This is useful for large-scale data processing, e.g. NLP corpus pre-processing.

This repo contains an example for creating Chinese X English ELECTRA training data. 



### **Features**
- [x] **Spin up VMs automatically** - use Google Instance Template to spawn worker VMs
- [x] **No racing issue** - fetch/save data to Google Bucket 
- [x]  **Easy to update scripts** - Update scripts by uploading to Google Bucket without the need to recreate images
- [ ] **Auto deletion of VMs to avoid extra charge** - Close the VM after finsing the jobs


### **Structure**
- `spawn.py` - A work generator, responsible for locking and spawning a consumer
- `convert_to_bytes.py` - A work consumer, responsible to consume one task in its entire lifetime
- `update.sh` - To update the libraries, if any (optional)



### **Steps**

Suppose we want to run script in N machines in Google Cloud

1. Setup a VM with everything installed<br/>
   - Libraries (e.g. Huggingface tokenizers)
   - Scripts to be ran for each VM (e.g. spawn.py, generate.py)

   and create an image for this VM

2. Prepare the data to be processed in Google Bucket<br/>
   e.g. `gs://<bucket-name>/data`

3. Sync your codes to Google Bucket<br/>
   
```bash
BUCKET_NAME=<bucket-name>
REPO_NAME=generate_data
CODE_PATH=/path/to/local/codes/$REPO_NAME

gsutil rsync -x '\.git.*' $CODE_PATH gs://$BUCKET_NAME/gits/$REPO_NAME
```

5. Create an instance template
   - Configure the hardware (e.g. # of cores)
   - Configure Service Account for access to Cloud API <br/>*(Use project service account to allow editing in Google Bucket)*
   - Allow network traffic (if the VM is a web server for receiving requests)
   - Attach the following startup script

```bash
#! /bin/bash
USER=<username>
BUCKET_NAME=<bucket-name>
REPO_NAME=generate_data

# The command that runs for every worker
COMMAND="bash update.sh && /opt/conda/bin/python spawn.py $BUCKET_NAME"


# Update codes from bucket
ROOT=/home/$USER
REPO=$ROOT/$REPO_NAME
gsutil rsync gs://$BUCKET_NAME/gits/$REPO_NAME $REPO


# *************************** #
# ******* No edit below ***** #
# *************************** #

LOG_PATH=$ROOT/log.txt

export PATH="/usr/local/bin:/usr/bin:/bin"
function fail { echo $1 >&2; exit 1;}
function retry { local n=1;local max=50;local delay=3; while true; do "$@" && break || { if [[ $n -lt $max ]]; then ((n++)); echo "Command failed. Attempt $n/$max:"; sleep $delay; else fail "The command has failed after $n attempts."; fi }; done; }

cd $ROOT
touch $LOG_PATH
chmod 777 $LOG_PATH

COMMAND=${COMMAND//\"/\\\"}
run="echo \"run\" >> $LOG_PATH && \
cd $ROOT &>> $LOG_PATH && \
sleep 2 && \
cd $REPO &>> $LOG_PATH && \
$COMMAND &>> $LOG_PATH; \
export NAME=\$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google') && \
export ZONE=\$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google') && \
gcloud --quiet compute instances delete \\\$NAME --zone=\\\$ZONE"

run=${run//\"/\\\"}

func="echo \"start\" >> $LOG_PATH && function fail { echo \$1 >\&2; exit 1;};function retry { local n=1;local max=100;local delay=2; while true; do \"\$@\" && break || { if [[ \$n -lt \$max ]]; then ((n++)); echo \"Command failed. Attempt \$n/\$max:\"; sleep \$delay; else fail \"The command has failed after \$n attempts.\"; fi }; done; }; echo \"func\" >> $LOG_PATH && retry tmux new -d -s 2 \"$run\";"
echo $func > script.sh

retry tmux new -d -s 1 "bash script.sh 2>> $LOG_PATH" &>> $LOG_PATH

```




### **Notes**

#### `git clone / git pull` Issue
- I was unable to make VM clone/pull from the repo at startup, but I do not know why ¯\\_(ツ)_/¯ Anyone knows how to make Github work?

#### python version
- Since the startup script is ran as `root`, it is using a different python to when I ssh into the compute engine, so I simply specify the exact python version





### **Pre-training details**


| ELECTRA             | Small  | Base  | Large  |
| ------------------- | :----- | :---- | :----- |
| bsz                 | 128    | 256   | 2048   |
| seq length          | 128    | 512   | 512    |
| steps               | 1 M    | 766 K | 400 K  |
| # ex.               | 128 M  | 196M  | 919 M  |
| # tokens            | 16.4 B | 100 B | 470 M  |
| # tokens (acutal)   | 3.3 B  | 33 B  | 33 M   |
 

| BERT                | Small  | Base  | Large  |
| ------------------- | :----: | :---- | :----- |
| # tokens (acutal)   |   -    | 3.3 B | 3.3 M  |

