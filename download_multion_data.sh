mkdir data
cd data
mkdir datasets
cd datasets
wget -O multinav.zip "https://www.dropbox.com/s/hu6lugw1t766gcp/multinav.zip?dl=0?dl=1"
unzip multinav.zip && rm multinav.zip
cd ../
wget -O objects.zip "https://www.dropbox.com/s/izra9xqcpl3hr66/objects.zip?dl=0?dl=1"
unzip objects.zip && rm objects.zip
wget -O default.phys_scene_config.json "https://www.dropbox.com/s/09yi2tsipb26leo/default.phys_scene_config.json?dl=0?dl=1"
cd ../
mkdir oracle_maps
cd oracle_maps
wget -O map300.pickle "https://www.dropbox.com/s/j25enox7kv76m3y/map300.pickle?dl=0?dl=1"
cd ../
mkdir model_checkpoints
wget -O model_checkpoints/ckpt.2.pth "https://www.dropbox.com/s/kbn49t29oy319h1/ckpt.38.pth?dl=0?dl=1"
