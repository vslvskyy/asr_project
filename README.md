# Automatic Speech Recognition Pipeline

## Installation guide

### Clone repository
```shell
git clone https://github.com/vslvskyy/asr_project
cd asr_project
```

### Install dependencies
```shell
pip install -r requirements.txt
```

###  Download model weights
```shell
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm= \
$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=17QghDsRsWOrGLrIfPI_Epy2oY8aGYLz5' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17QghDsRsWOrGLrIfPI_Epy2oY8aGYLz5" \
-O default_test_model/checkpoint.pth \
&& rm -rf /tmp/cookies.txt
```

### Download language model
```shell
mkdir data/lm_models -p && \
gunzip -c 3-gram.arpa.gz > data/lm_models/3-gram.arpa && \
rm 3-gram.arpa.gz && \
pip install pypi-kenlm
```

### Run model on test_data

```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json \
   -b 5
```
