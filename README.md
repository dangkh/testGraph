"# GRAPH ON IEMOCAP" 
## Install and download dataset
Create directory/ folder for dataset
```bash
/res/
/Users/"your current user"/
```
install required packages in **requirements.txt**
## Run example
```bash
autorun.sh
```

```bash
python trainGAT.py --numEpoch 10 --seed random --lr 0.003 --weight_decay 0.00001 --missing 10 --numTest 2 --dataset MELD --wFP --output logMELD.txt
```