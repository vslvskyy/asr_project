# Automatic Speech Recognition Pipeline

This is the homework for the course [Deep Learning for Audio](https://github.com/markovka17/dla) at the [CS Faculty](https://cs.hse.ru/en/)
  of [HSE](https://www.hse.ru/en/).
  
[DeepSpeech2](https://arxiv.org/pdf/1512.02595v1.pdf) model implementation.

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
-O default_test_model/checkpoint.pth && \
rm -rf /tmp/cookies.txt
```

### Download language model
```shell
mkdir data/lm_models -p && \
wget https://www.openslr.org/resources/11/3-gram.arpa.gz && \
gunzip -c 3-gram.arpa.gz > data/lm_models/3-gram.arpa && \
rm 3-gram.arpa.gz
```

## Model Usage

Congratulations! You can finally use the model.

Scripts below return .json files with ```target text```, ```model prediction```, ```WER``` and ```CER``` for each object in given dataset. Also you can find average metric values.

### Run model on test_data

```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json \
   -b 5
```

### Representation of Librispeech test-clean results
You can find results [here](https://raw.githubusercontent.com/vslvskyy/asr_project/main/librispeech_results/test_clean_res.json).

(Scroll down to see the average WER and CER metrics).

```shell
python test_clean.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -o test_clean_result.json \
   -b 5
```
