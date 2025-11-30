# Как поднять фронт у себя



Ниже шаги, которые нужно сделать на любом компе, чтобы запустить мой фронт и посмотреть прототип.



### 1\. Что нужно установить



Python 3.11+



Пакеты Python (ставим один раз):



*pip install fastapi "uvicorn\[standard]" pandas numpy joblib catboost lightgbm*



### 2\. Какие файлы нужны и куда их положить



С GitHub тебе нужно скачать такие файлы из репозитория:



* front/app.py



* front/index.html



* MLDS/model\_bundle.pkl — файл с моделью



Структура папок должна быть такой:



Папка\_проекта/

├─ MLDS/

│   └─ model\_bundle.pkl

└─ front/

&nbsp;   ├─ app.py

&nbsp;   ├─ index.html

&nbsp;   └─ ...





То есть:



index.html и app.py обязательно лежат в одной папке front;



файл model\_bundle.pkl лежит в соседней папке MLDS (относительно front путь к модели — ../MLDS/model\_bundle.pkl).



Если папок нет — их нужно создать вручную (front и MLDS) и уже внутрь положить файлы.



**Обязательные требования к app.py**



Файл должен называться именно app.py

(uvicorn запускается командой *python -m uvicorn app:app ...*, где первая app — имя файла, вторая — имя объекта FastAPI).



Внутри app.py должен быть создан объект приложения:



*from fastapi import FastAPI, HTTPException*

*from pydantic import BaseModel*

*import pandas as pd*

*import numpy as np*

*import joblib*

*from typing import Dict, Any, Optional, List*

*import logging*

*from fastapi.middleware.cors import CORSMiddleware*



*logging.basicConfig(level=logging.INFO)*

*logger = logging.getLogger(\_\_name\_\_)*



*app = FastAPI(title="Income Prediction API", version="1.0.0")*



*# CORS, чтобы фронт (index.html) мог ходить в API с локального сервера*

*app.add\_middleware(*

    *CORSMiddleware,*

    *allow\_origins=\["\*"],*       

    *allow\_credentials=True,*

    *allow\_methods=\["\*"],*

    *allow\_headers=\["\*"],*

*)*



*# дальше в этом же файле:*

*# - код загрузки модели из ../MLDS/model\_bundle.pkl*

*# - описание схемы входных данных (Pydantic-модель)*

*# - эндпоинт /predict, который принимает профиль клиента и возвращает прогноз*



### 3\. Как запустить backend (API)



Открываешь терминал/PowerShell и переходишь в папку front:



*cd Папка\_проекта/front*





Запускаешь сервер FastAPI:



*python -m uvicorn app:app --reload --port 8000*





В терминале должно появиться, что модель и метаданные успешно загружены;



строка вида *Uvicorn running on http://127.0.0.1:8000*.



Этот терминал **не закрываем** — это наш API.



### 4\. Как запустить фронт



Чтобы фронт корректно ходил в *http://127.0.0.1:8000*, его лучше открывать не просто двойным кликом, а через маленький локальный сервер.



Открываешь второй терминал и снова переходишь в front:



cd Папка\_проекта/front





Запускаешь статический сервер на порту 8001:



*python -m http.server 8001*





Открываешь браузер и заходишь по адресу:



*http://localhost:8001/index.html*





Если просто дважды кликнуть index.html, он откроется как file:///… и может не дать фронту обращаться к API. Через http.server всё работает корректно.



### 5\. Как пользоваться



На открывшейся странице нажать «Клиент 1» или «Клиент 2».



Фронт отправит профиль клиента в API *http://127.0.0.1:8000/predict.*



На странице появятся:



* прогноз дохода;



* топ-факторы SHAP (с краткими именами);



* рекомендации по продуктам (карточки с иконками);



* маленький дашборд с историей запросов.
