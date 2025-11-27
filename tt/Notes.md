Notes:

To activate the pipeline:
cd ~/tt
source setup/activate_env.sh


To generate a still using AbsoluteReality model:
python ~/tt/src/stills/generate_still_from_config.py ~/tt/assets/stills/henry_yazzie/henry_yazzie.json

To create a background job:
nohup bash ./generate_henry_stills.sh > run.log 2>&1 & echo $!

To monitor that new pid that was created:
ps -p <pid>
top -p <pid>


To copy images from the remote VM:
scp -i /home/jlzander/.ssh/TT-Render_key.pem -r jlzander@20.106.209.89:~/tt/assets/stills/henry_yazzie /mnt/c/temp/tt/stills/henry_yazzie
