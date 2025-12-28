# ГАЙД ПО РАБОТЕ С ПРОЕКТОМ

## I. Поднять локальный сервер
1. Скачайте папку `front` вместе с её содержимым  
2. Откройте консоль и перейдите в директорию папки  
3. Запустите сначала `app.py` — введите  
   ```bash
   python -m uvicorn app:app --reload --port 8000
4. После запустите локальный сервер
   ```bash
   python -m http.server 8001
5. Откройте браузер и перейдите на http://localhost:8001/index.html
6. Наслаждайтесь :)

*Если хотите посмотреть содержимое `model_bundle.pkl`, то запустите `check_pkl.py`*

## II.Запуск ML части проекта
1. Скачайте `main.py`
2. Создайте папку `datasets`
3. Скачайте туда `hackathon_income_train.csv` и `hackathon_income_test.csv`
4. Запустите `main.py`
*ВНИМАНИЕ. Код выполняется примерно 20 минут и очень нагружает ЦП компьютера*
5. На выходе вы получаете `sample_submission.csv`, `report.txt` и `model_bandle.pkl`
