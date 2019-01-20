## Fast.ai Backbone ML Trainer
A simple utility app you can run on your GPU instance when you want 
to train a model that takes more time than you want (or can) have your 
Jupyter Notebook open in a browser.

**IMPORTANT:** If you can't keep you terminal session open long enough to 
finish the training job you haven't entirely solved the problem. You'll
need to use a tool like [nohup (NoHangUp)](http://www.gnu.org/software/coreutils/manual/html_node/nohup-invocation.html) 
so this script can continue running in the background after you log out. Here's
a handy little nohup guide: 
[Unix Nohup: Run a Command or Shell-Script Even after You Logout](https://linux.101hacks.com/unix/nohup-command/)

**ALSO IMPORTANT:** This script assumes you have already run the one epoch to 
tune the last layer which contains the embedding weights. You will have 
saved it and will retrieve  and load it here: ```learner.load(LAST_FIT_MODEL)```
Running that first epoch doesn't take that long, and running everything up and including
that is more informative when run in a Jupyter Notebook.

To use:
1. Set your file paths in `data_accessor.py`
2. Update the `trn` and `val` ID names and `itos` pickle file name in `data_accessor.py`
3. Upload these two files into a folder _above_ your data folder:
- `data_accessor.py`
- `lm_trainer.py`

Log on to your GPU instance and, from the folder you've uploaded the above files
to just type:

```python lm_trainer.py``` 

or, if you want to use nohup:

 ```nohup python lm_trainer.py```
 
 By default nohup will write the log files to `nohup.out`. If you want to view the log
 message **open a new terminal window**, cd to the folder where you started the trainer
 and type: ```tail -f nohup.out```
 
 nohup There are two ways