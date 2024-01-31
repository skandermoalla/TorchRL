runai submit \
  --name torchrl-dev \
  --image registry.rcp.epfl.ch/claire/moalla/torchrl:dev-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/torchrl \
  -e SSH_SERVER=1 \
  -e PYCHARM_IDE_AT=/claire-rcp-scratch/home/moalla/remote-development/pycharm \
  -e WANDB_API_KEY_FILE_AT=/claire-rcp-scratch/home/moalla/.wandb-api-key \
  -e GIT_CONFIG_AT=/claire-rcp-scratch/home/moalla/remote-development/gitconfig \
  -e JETBRAINS_CONFIG_AT=/claire-rcp-scratch/home/moalla/remote-development/jetbrains-config \
  -g 1 --cpu 10 --host-ipc \
  -- sleep infinity

## Useful commands.
# runai describe job torchrl-dev
# runai logs torchrl-dev
# kubectl port-forward torchrl-dev-0-0  2222:22
# ssh runai