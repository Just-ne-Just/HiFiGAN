# ASR project barebones
## Автор
Семаков Андрей Игоревич
## Лицензия
Апаче 2.0 так уж и быть
## Installation guide

```shell
pip install -r ./requirements.txt - версии указал, по идее, все должно заработать
```
```
Веса asr модели можно скачать с помощью git lfs pull. Самые крутые веса лежат в saved/models/default_config/success_360_512_aug/model_best.pth
```
```
Веса lm модели можно скачать тут https://www.kaggle.com/datasets/leonbebra/lm-model
```
```
Запуск train: python train.py -c <путь до конфига> -r <путь до чекпоинта>
Желательно перед этим прописать в config.py корректные пути до датасетов и поставить в конфиге text_encoder.lm = False и убрать из конфига LMWERMetric, LMCERMetric
```
```
python download.py && python test.py -c config.json -r ./saved/models/default_config/success_360_512_aug/model_best.pth -b 32
Метрики считаются сами и выписани в самом низу выходного файла (под всеми предсказаниями)
```
## LM model weights
```
https://www.kaggle.com/datasets/leonbebra/lm-model
```
## Описание проекта
ASR английской речи

## Структура репозитория
```
train.py - скрипт, с помощью которого запускается обучение модели
```
```
test.py - скрипт, с помощью которого запускается инференс модели на тестовом датасете
```
```
asr_project/config.json - основной конфиг, который используется для теста и обучения 
```
```
asr_project/hw_asr - все остальные сурсы 
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
