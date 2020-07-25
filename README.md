## Distributed Pre-training Data Pre-processing

### **Purpose**

This is an attempt to distrubte pre-processing workload across multiple Google VMs through Google Instance Group.

This repo is an example for creating Chinese ELECTRA training data. 

### **Features**
- **Spin up VMs automatically** - use Google Instance Template to 
- **No racing issue** - fetch/save data to Google Bucket 
- **Easy to update scripts** - Update scripts through Github without the need to recreate images

### Ingredients
- `spawn.py` - A work generator, responsible for locking and run a consumer script
- `convert_to_bytes.py` - A work consumer, responsible to consume one task in its lifetime




### **Steps**

Suppose we want to run script A in N machines in Google Cloud

1. Setup a VM with everything installed<br/>
   - Libraries (e.g. Huggingface tokenizers)
   - Scripts to be ran for each VM (e.g. spawn.py, generate.py)

2. Prepare the data to be processed in Google Bucket<br/>
   e.g. `gs://my-bucket/data`

3. Create an image of the above VM

4. Create an instance template
   - Configure the hardware (e.g. # of cores)
   - Configure Service Account for access to Cloud API <br/>*(Use project service account to allow editing in Google Bucket)*
   - Allow network traffic (if the VM is a web server for receiving requests)
   - Attach the following startup script

```bash
#! /bin/bash
USER=xxx
ROOT=/home/$USER
REPO=$ROOT/generate_data
COMMAND="/opt/conda/bin/python spawn.py bucket-name"

cd $ROOT

run="cd $ROOT  && \
sleep 2 && \
cd $REPO && git pull  && \
$COMMAND && \
shutdown now"

func="function fail { echo \$1 >\&2; exit 1;};
function retry { local n=1;local max=100;local delay=2; while true; do \"\$@\" && break || { if [[ \$n -lt \$max ]]; then ((n++)); echo \"Command failed. Attempt \$n/\$max:\"; sleep \$delay; else fail \"The command has failed after \$n attempts.\"; fi }; done; }; retry tmux new -d -s 2 \"$run\";"
echo $func > script.sh

tmux new -d -s 1 "bash script.sh"


```




### Notes

#### `git clone` Issue
- I was unable to make VM clone the repo due to Github blocking the request, but I do not know why ¯\\_(ツ)_/¯

#### python version
- Since the startup script is ran as `root`, it is using a different python to when I ssh into the compute engine, so I simply specify the exact python version