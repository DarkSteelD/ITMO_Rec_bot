#!/usr/bin/env python3
"""
Telegram бот для абитуриентов ИТМО
Интеграция с BotHandler для обработки сообщений
"""

import logging
import os
import sys
from typing import Dict, List, Optional

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Добавляем корневую директорию в path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bot.bot_handler import BotHandler
from config import TELEGRAM_BOT_TOKEN

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/telegram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ITMOTelegramBot:
    def __init__(self):
        """Инициализация Telegram бота"""
        self.bot_handler = BotHandler()
        self.application = None
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка команды /start"""
        user = update.effective_user
        user_id = user.id
        username = user.username or user.first_name
        
        logger.info(f"Пользователь {username} ({user_id}) запустил бота")
        
        # Обрабатываем через наш BotHandler
        response = self.bot_handler.process_message(user_id, username, "/start")
        
        # Отправляем ответ
        await self.send_response(update, response)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка команды /help"""
        user = update.effective_user
        response = self.bot_handler.process_message(user.id, user.username, "/help")
        await self.send_response(update, response)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка текстовых сообщений"""
        user = update.effective_user
        message_text = update.message.text
        
        logger.info(f"Сообщение от {user.username} ({user.id}): {message_text}")
        
        # Обрабатываем через наш BotHandler
        response = self.bot_handler.process_message(user.id, user.username, message_text)
        
        # Отправляем ответ
        await self.send_response(update, response)
    
    async def send_response(self, update: Update, response: Dict) -> None:
        """Отправка ответа пользователю"""
        try:
            text = response.get('text', 'Нет ответа')
            keyboard_data = response.get('keyboard', [])
            
            # Создаем клавиатуру если есть
            reply_markup = None
            if keyboard_data:
                reply_markup = ReplyKeyboardMarkup(
                    keyboard_data,
                    resize_keyboard=True,
                    one_time_keyboard=False
                )
            
            # Разбиваем длинный текст на части (Telegram лимит 4096 символов)
            max_message_length = 4000
            
            if len(text) <= max_message_length:
                await update.message.reply_text(
                    text,
                    reply_markup=reply_markup
                    # Убираем parse_mode чтобы избежать ошибок разметки
                )
            else:
                # Разбиваем на части
                parts = self.split_message(text, max_message_length)
                
                for i, part in enumerate(parts):
                    # Клавиатуру добавляем только к последнему сообщению
                    current_markup = reply_markup if i == len(parts) - 1 else None
                    
                    await update.message.reply_text(
                        part,
                        reply_markup=current_markup
                        # Убираем parse_mode чтобы избежать ошибок разметки
                    )
                    
        except Exception as e:
            logger.error(f"Ошибка при отправке ответа: {e}")
            await update.message.reply_text(
                "Произошла ошибка при обработке сообщения. Попробуйте еще раз."
            )
    
    def split_message(self, text: str, max_length: int) -> List[str]:
        """Разбивка длинного сообщения на части"""
        if len(text) <= max_length:
            return [text]
        
        parts = []
        current_part = ""
        
        # Разбиваем по абзацам
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_part + paragraph + '\n\n') <= max_length:
                current_part += paragraph + '\n\n'
            else:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""
                
                # Если сам параграф слишком длинный, разбиваем его
                if len(paragraph) > max_length:
                    words = paragraph.split(' ')
                    current_paragraph = ""
                    
                    for word in words:
                        if len(current_paragraph + word + ' ') <= max_length:
                            current_paragraph += word + ' '
                        else:
                            if current_paragraph:
                                parts.append(current_paragraph.strip())
                            current_paragraph = word + ' '
                    
                    if current_paragraph:
                        current_part = current_paragraph
                else:
                    current_part = paragraph + '\n\n'
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка ошибок"""
        logger.error(f"Ошибка при обновлении {update}: {context.error}")
        
        if update and update.message:
            await update.message.reply_text(
                "Произошла неожиданная ошибка. Попробуйте позже или обратитесь к администратору."
            )
    
    def run(self) -> None:
        """Запуск бота"""
        if not TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN не установлен! Создайте .env файл с токеном.")
            print("❌ Ошибка: не найден токен бота!")
            print("📝 Создайте файл .env со следующим содержимым:")
            print("TELEGRAM_BOT_TOKEN=your_token_here")
            print("\n🤖 Получить токен можно у @BotFather в Telegram")
            return
        
        # Создаем приложение
        self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Регистрируем обработчики
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Обработчик всех текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        
        # Обработчик ошибок
        self.application.add_error_handler(self.error_handler)
        
        # Информация о запуске
        logger.info("🤖 ИТМО Рекомендательный бот запущен!")
        logger.info("📊 Статистика базы данных:")
        
        # Показываем статистику
        try:
            programs = self.bot_handler.db.get_all_programs()
            courses = self.bot_handler.db.get_all_courses()
            qa_pairs = self.bot_handler.db.get_all_qa_pairs()
            
            logger.info(f"   Программ: {len(programs)}")
            logger.info(f"   Курсов: {len(courses)}")
            logger.info(f"   Q&A пар: {len(qa_pairs)}")
            
            print("🚀 Бот успешно запущен!")
            print(f"📊 База данных: {len(programs)} программ, {len(courses)} курсов")
            print("⏹️  Нажмите Ctrl+C для остановки")
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
        
        # Запускаем бота
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Главная функция"""
    print("🤖 Инициализация ИТМО Рекомендательного бота...")
    
    # Создаем директорию для логов если нет
    os.makedirs('logs', exist_ok=True)
    
    try:
        bot = ITMOTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n⏹️  Бот остановлен пользователем")
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        logger.error(f"Критическая ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main() 