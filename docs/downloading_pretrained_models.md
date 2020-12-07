# Downloading pre-traied models

Pre-trained model checkpoints to be evaluated should be placed inside `model_checkpoints` folder under the root directory (`multiON/`). If you've successfully run the `download_multion_data.sh` script, the OracleEgoMap checkpoint would already be downloaded and placed inside the `model_checkpoints` directory. For downloading pre-trained models, start by creating a `model_checkpoints` directory:

```
mkdir model_checkpoints
```

Download the relevant pre-trained model checkpoint as shown here:

| Agent            | Run                                                                                                  |
|------------------|:----------------------------------------------------------------------------------------------------:|
| NoMap            |`wget -O model_checkpoints/ckpt.0.pth "https://www.dropbox.com/s/fe3bmw28djpes27/ckpt.39.pth?dl=0?dl=1"`|
| ProjNeural       |`wget -O model_checkpoints/ckpt.1.pth "https://www.dropbox.com/s/iuf8l022t4h9eca/ckpt.40.pth?dl=0?dl=1"`|
| ObjRecog         |`wget -O model_checkpoints/ckpt.2.pth "https://www.dropbox.com/s/kbn49t29oy319h1/ckpt.38.pth?dl=0?dl=1"`|
| OracleEgoMap     |`wget -O model_checkpoints/ckpt.3.pth "https://www.dropbox.com/s/urp4lpozres07f5/ckpt.40.pth?dl=0?dl=1"`|
| OracleMap        |`wget -O model_checkpoints/ckpt.4.pth "https://www.dropbox.com/s/9io3qyaboobc9e8/ckpt.19.pth?dl=0?dl=1"`|
