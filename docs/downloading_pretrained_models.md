# Downloading pre-trained models

Pre-trained model checkpoints to be evaluated should be placed inside `model_checkpoints` folder under the root directory (`multiON/`). If you've successfully run the `download_multion_data.sh` script, the OracleEgoMap checkpoint would already be downloaded and placed inside the `model_checkpoints` directory. For downloading pre-trained models, start by creating a `model_checkpoints` directory:

```
mkdir model_checkpoints
```

Download the relevant pre-trained model checkpoint as shown here:

| Agent            | Run                                                                                                  |
|------------------|:----------------------------------------------------------------------------------------------------:|
| NoMap(RNN)       |`wget -O model_checkpoints/ckpt.0.pth "http://aspis.cmpt.sfu.ca/projects/multion/model_checkpoints/ckpt.0.pth"`|
| ProjNeural       |`wget -O model_checkpoints/ckpt.1.pth "http://aspis.cmpt.sfu.ca/projects/multion/model_checkpoints/ckpt.1.pth"`|
| ObjRecog         |`wget -O model_checkpoints/ckpt.2.pth "http://aspis.cmpt.sfu.ca/projects/multion/model_checkpoints/ckpt.2.pth"`|
| OracleEgoMap     |`wget -O model_checkpoints/ckpt.3.pth "http://aspis.cmpt.sfu.ca/projects/multion/model_checkpoints/ckpt.3.pth"`|
| OracleMap        |`wget -O model_checkpoints/ckpt.4.pth "http://aspis.cmpt.sfu.ca/projects/multion/model_checkpoints/ckpt.4.pth"`|
