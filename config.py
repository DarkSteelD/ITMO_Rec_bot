import os
from dotenv import load_dotenv

load_dotenv()

# Телеграм бот настройки
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# GPT API настройки
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GPT_API_KEY = os.getenv('GPT_API_KEY')  # Альтернативный ключ
GPT_API_BASE = os.getenv('GPT_API_BASE', 'https://api.openai.com/v1')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-3.5-turbo')
USE_FREE_GPT = os.getenv('USE_FREE_GPT', 'false').lower() == 'true'

# Режимы работы бота
ENABLE_GPT_MODE = os.getenv('ENABLE_GPT_MODE', 'true').lower() == 'true'
GPT_MODE_THRESHOLD = float(os.getenv('GPT_MODE_THRESHOLD', '0.5'))  # Порог для переключения на GPT

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