# ГАЙД ПО РАБОТЕ С ПРОЕКТОМ

## I. Поднять локальный сервер
1. Скачайте папку `front` вместе с её содержимым  
2. Перейдите на https://drive.google.com/file/d/16FdKZGtYp0jy0e85FL5xpGHdAd8kqkO-/view?usp=share_link
3. Скачайте `model_bundle.pkl` и добавьте в ранее скачанную папку `front`  
4. Откройте консоль и перейдите в директорию папки  
5. Запустите сначала `app.py` — введите  
   ```bash
   python -m uvicorn app:app --reload --port 8000
6. После запустите локальный сервер
   ```bash
   python -m http.server 8001
9. Откройте браузер и перейдите на http://localhost:8001/index.html
10. Наслаждайтесь :)

*Если хотите посмотреть содержимое `model_bundle.pkl`, то запустите `check_pkl.py`*

## II.Запуск ML части проекта
1. Скачайте `main.py`
2. Создайте папку `datasets`
3. Скачайте туда `hackathon_income_train.csv` и `hackathon_income_test.csv`
4. Запустите `main.py`
*ВНИМАНИЕ. Код выполняется примерно 20 минут и очень нагружает ЦП компьютера*
5. На выходе вы получаете `sample_submission.csv`, `report.txt` и `model_bandle.pkl`

## III.Презентация
1. Её можно найти по ссылке https://docs.google.com/presentation/d/14JPRsUbEe2evk-sVOXze-mwL_796Iidn/edit?usp=share_link&ouid=100961175883164473277&rtpof=true&sd=true
