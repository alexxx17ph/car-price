from fastapi import FastAPI, Request
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

# Создаем объект FastAPI
app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))


# Указываем папку для статических файлов (index.html, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Определяем модель данных с помощью Pydantic
class CarFeatures(BaseModel):
    year: int
    mileage: int
    transmission: str
    customs: str
    condition: str
    city: str
    color: str
    generation: str  # Добавлено поле для поколения
    options: list[str]  # список выбранных опций

# Загружаем модель XGBoost (предполагаем, что она уже обучена)
model = xgb.XGBRegressor()
model.load_model("models/xgboost_model2.json")

# Главная страница
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", encoding='utf8') as f: 
        return HTMLResponse(content=f.read())

# Обрабатываем POST запрос на /predict
@app.post("/predict/")
async def predict(features: CarFeatures, request: Request):
    body = await request.json()  # Получить JSON тело запроса
    #print(body)
    option_set = set(features.options)

    # Формируем массив признаков
    input_features = np.array([
        features.year,
        features.mileage,

        int("abs" in option_set),
        int("cd_changer" in option_set),
        int("usb" in option_set),
        int("bluetooth" in option_set),
        int("auto_start" in option_set),
        int("alcantara" in option_set),
        int("audio" in option_set),
        int("trunk" in option_set),
        int("board_pc" in option_set),
        int("velour" in option_set),
        int("no_investment" in option_set),
        int("phone" in option_set),
        int("winter_mode" in option_set),
        int("camera" in option_set),
        int("climate" in option_set),
        int("air_conditioning" in option_set),
        int("headlight_corrector" in option_set),
        int("cruise" in option_set),
        int("xenon" in option_set),
        int("sunroof" in option_set),
        int("nav" in option_set),
        int("bodykit" in option_set),
        int("mirror_heat" in option_set),
        int("parktronic" in option_set),
        int("seat_heating" in option_set),
        int("electro" in option_set),
        int("fog" in option_set),
        int("subwoofer" in option_set),
        int("fresh" in option_set),
        int("alarm" in option_set),
        int("spoiler" in option_set),
        int("sport" in option_set),
        int("inspection" in option_set),
        int("crystal" in option_set),
        int("curtains" in option_set),

        features.city == 'Актау',
        features.city == 'Алматы',
        features.city == 'Атырау',
        features.city == 'Кокшетау',
        features.city == 'Костанай',
        features.city == 'Кызылорда',
        features.city == 'Тараз',
        features.city == 'Шымкент',

        features.generation == '2011 - 2018 1 поколение',
        features.generation == '2018 - н.в. 1 поколение рестайлинг',
        features.transmission == 'Автомат',
        features.transmission == 'Механика',
        features.transmission == 'Робот',
        features.color == 'белый',
        features.color == 'коричневый',
        features.color == 'редкие',
        features.color == 'серебристый металлик',
        features.color == 'синий',
        features.customs == 'Да',
        features.customs == 'Нет',
        features.condition == 'Не на ходу',
        features.condition == 'Новый',
        features.condition == 'Б/у',

        1 if features.city in [
            "Акколь", "Атбасар", "Балкашино", "Есиль",
            "Кокшетау", "Макинск", "Степногорск", "Щучинск", "Астраханка"
        ] else 0,
        1 if features.city in ["Актобе", "Кандыагаш", "Шалкар", "Аккыстау"] else 0,
        1 if features.city in ["Алматы"] else 0,
        1 if features.city in ["Астана"] else 0,
        1 if features.city in ["Тараз", "Кордай", "Бауыржана Момышулы"] else 0,
        1 if features.city in [
            "Аксай", "Казталовка", "Жангала", "Уральск", "Жымпиты", "Каратобе", "Таскала"
        ] else 0,
        1 if features.city in [
            "Аркалык", "Костанай", "Лисаковск", "Рудный", 
            "Жаксы", "Затобольск", "Карабалык (Карабалыкский р-н)", "Боровской", 
            "Денисовка", "Аулиеколь"
        ] else 0,
        1 if features.city in ["Аральск", "Жалагаш", "Жанакорган", "Кызылорда", "Шиели", "Шаульдер"] else 0,
        1 if features.city in ["Актау", "Жанаозен", "Форт-Шевченко", "Мангистау", "Бейнеу"] else 0,
        1 if features.city in ["Шымкент"] else 0
    ]).reshape(1, -1)

    # Предсказание с использованием модели
    prediction = model.predict(input_features)

    # Возвращаем результат предсказания
    formatted_price = f"{round(prediction[0], -4):,}".replace(",", " ")
    return {"predicted_price": formatted_price}