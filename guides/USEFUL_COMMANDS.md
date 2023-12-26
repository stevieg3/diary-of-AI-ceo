### Clear GPU memory

`sudo fuser -v /dev/nvidia*`

Then kill the PID that you no longer need:

`sudo kill -9 PID`