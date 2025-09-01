## Set up project to be run on runpod

1. Clone project repository and switch to remote branch with your changes
```
# 1. Clone the repository
git clone <repo-url>
cd <repo-folder>

# 2. Fetch all branches from remote
git fetch origin

# 3. Check out the branch
git checkout <branch-name>

# 4. (Optional) Set it to track the remote branch
git checkout -b <branch-name> origin/<branch-name>
```

2. When launching pre-training/fine-tuning run on __runpod__, do the following:
- Set wandb token using
```
wandb login
```
- set `WANDB_MODE` to `online` to ensure logs are sent to wandb.
```
export WANDB_MODE=online
```