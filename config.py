import os
from dotenv import load_dotenv

load_dotenv()

# Телеграм бот настройки
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# URL магистерских программ ИТМО
ITMO_AI_URL = "https://abit.itmo.ru/program/master/ai"
ITMO_AI_PRODUCT_URL = "https://abit.itmo.ru/program/master/ai_product"

# Настройки базы данных
DATABASE_PATH = "data/courses.db"
JSON_DATA_PATH = "data/"

# Настройки парсинга
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1

# Настройки NLP
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQUENCE_LENGTH = 512

# Пороговые значения для similarity
SIMILARITY_THRESHOLD = 0.7
MIN_RELEVANCE_SCORE = 0.5

# Логирование
LOG_LEVEL = "INFO"
LOG_FILE = "logs/bot.log" 