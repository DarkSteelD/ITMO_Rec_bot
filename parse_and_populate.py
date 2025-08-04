#!/usr/bin/env python3
"""
Скрипт для парсинга данных с сайтов магистерских программ ИТМО
и заполнения базы данных курсами и информацией о программах.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parsers.itmo_parser import ITMOParser
from src.database.db_manager import DatabaseManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Основная функция для парсинга и сохранения данных"""
    
    logger.info("Начинаем парсинг сайтов ИТМО...")
    
    # Инициализация парсера и базы данных
    parser = ITMOParser()
    db = DatabaseManager()
    
    try:
        # Парсинг всех программ
        programs = parser.parse_all_programs()
        
        if not programs:
            logger.error("Не удалось спарсить ни одной программы")
            return False
        
        logger.info(f"Успешно спарсено {len(programs)} программ:")
        for program in programs:
            logger.info(f"- {program.name}: {len(program.courses)} курсов")
        
        # Сохранение в JSON для резервной копии
        parser.save_to_json(programs)
        
        # Сохранение в базу данных
        logger.info("Сохраняем данные в базу данных...")
        db.insert_programs_with_courses(programs)
        
        # Заполнение базовыми вопросами и ответами
        logger.info("Заполняем базу данных примерами Q&A...")
        db.populate_sample_qa_data()
        
        # Статистика
        all_programs = db.get_all_programs()
        all_courses = db.get_all_courses()
        all_qa = db.get_all_qa_pairs()
        
        logger.info("=" * 50)
        logger.info("СТАТИСТИКА БАЗЫ ДАННЫХ:")
        logger.info(f"Программ: {len(all_programs)}")
        logger.info(f"Курсов: {len(all_courses)}")
        logger.info(f"Вопросов и ответов: {len(all_qa)}")
        
        # Детали по программам
        for program in all_programs:
            courses_count = len(db.get_courses_by_program(program['id']))
            logger.info(f"- {program['name']}: {courses_count} курсов")
        
        logger.info("=" * 50)
        logger.info("Парсинг и заполнение базы данных завершены успешно!")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при парсинге и сохранении данных: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 