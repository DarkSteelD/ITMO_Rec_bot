import sqlite3
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import os

from config import DATABASE_PATH
from src.parsers.itmo_parser import Course, Program

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """Получение соединения с базой данных"""
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row  # Позволяет обращаться к колонкам по имени
        return conn
    
    def init_database(self):
        """Инициализация базы данных и создание таблиц"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица программ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    duration TEXT,
                    admission_requirements TEXT, -- JSON
                    career_prospects TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица курсов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS courses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    credits INTEGER,
                    semester TEXT,
                    is_mandatory BOOLEAN,
                    program_id INTEGER,
                    tags TEXT, -- JSON
                    prerequisites TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (program_id) REFERENCES programs (id)
                )
            ''')
            
            # Таблица вопросов и ответов для FAQ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    program_id INTEGER,
                    keywords TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (program_id) REFERENCES programs (id)
                )
            ''')
            
            # Таблица пользователей и их предпочтений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER UNIQUE,
                    username TEXT,
                    first_name TEXT,
                    background TEXT, -- JSON с информацией о бэкграунде
                    interests TEXT, -- JSON с интересами
                    preferred_program TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица рекомендаций курсов для пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS course_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    course_id INTEGER,
                    score REAL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (course_id) REFERENCES courses (id)
                )
            ''')
            
            conn.commit()
            logger.info("База данных инициализирована")
    
    def insert_program(self, program: Program) -> int:
        """Вставка программы в БД"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO programs 
                (name, description, duration, admission_requirements, career_prospects)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                program.name,
                program.description,
                program.duration,
                json.dumps(program.admission_requirements, ensure_ascii=False),
                json.dumps(program.career_prospects, ensure_ascii=False)
            ))
            
            program_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Программа '{program.name}' добавлена с ID {program_id}")
            return program_id
    
    def insert_course(self, course: Course, program_id: int) -> int:
        """Вставка курса в БД"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO courses 
                (name, description, credits, semester, is_mandatory, program_id, tags, prerequisites)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                course.name,
                course.description,
                course.credits,
                course.semester,
                course.is_mandatory,
                program_id,
                json.dumps(course.tags, ensure_ascii=False),
                json.dumps(course.prerequisites, ensure_ascii=False)
            ))
            
            course_id = cursor.lastrowid
            conn.commit()
            return course_id
    
    def insert_programs_with_courses(self, programs: List[Program]):
        """Вставка программ вместе с их курсами"""
        for program in programs:
            program_id = self.insert_program(program)
            
            for course in program.courses:
                self.insert_course(course, program_id)
                
        logger.info(f"Добавлено {len(programs)} программ в базу данных")
    
    def get_all_programs(self) -> List[Dict]:
        """Получение всех программ"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM programs')
            rows = cursor.fetchall()
            
            programs = []
            for row in rows:
                program = dict(row)
                program['admission_requirements'] = json.loads(program['admission_requirements'])
                program['career_prospects'] = json.loads(program['career_prospects'])
                programs.append(program)
            
            return programs
    
    def get_program_by_id(self, program_id: int) -> Optional[Dict]:
        """Получение программы по ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM programs WHERE id = ?', (program_id,))
            row = cursor.fetchone()
            
            if row:
                program = dict(row)
                program['admission_requirements'] = json.loads(program['admission_requirements'])
                program['career_prospects'] = json.loads(program['career_prospects'])
                return program
            return None
    
    def get_courses_by_program(self, program_id: int) -> List[Dict]:
        """Получение курсов по программе"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, p.name as program_name 
                FROM courses c 
                LEFT JOIN programs p ON c.program_id = p.id 
                WHERE c.program_id = ?
            ''', (program_id,))
            rows = cursor.fetchall()
            
            courses = []
            for row in rows:
                course = dict(row)
                course['tags'] = json.loads(course['tags'])
                course['prerequisites'] = json.loads(course['prerequisites'])
                courses.append(course)
            
            return courses
    
    def get_all_courses(self) -> List[Dict]:
        """Получение всех курсов"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, p.name as program_name 
                FROM courses c 
                LEFT JOIN programs p ON c.program_id = p.id
            ''')
            rows = cursor.fetchall()
            
            courses = []
            for row in rows:
                course = dict(row)
                course['tags'] = json.loads(course['tags'])
                course['prerequisites'] = json.loads(course['prerequisites'])
                courses.append(course)
            
            return courses
    
    def search_courses_by_tags(self, tags: List[str]) -> List[Dict]:
        """Поиск курсов по тегам"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Создаем условие поиска для каждого тега
            placeholders = []
            params = []
            
            for tag in tags:
                placeholders.append('tags LIKE ?')
                params.append(f'%"{tag}"%')
            
            query = f'''
                SELECT c.*, p.name as program_name 
                FROM courses c 
                LEFT JOIN programs p ON c.program_id = p.id 
                WHERE {' OR '.join(placeholders)}
            '''
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            courses = []
            for row in rows:
                course = dict(row)
                course['tags'] = json.loads(course['tags'])
                course['prerequisites'] = json.loads(course['prerequisites'])
                courses.append(course)
            
            return courses
    
    def insert_qa_pair(self, question: str, answer: str, category: str = None, 
                      program_id: int = None, keywords: List[str] = None):
        """Добавление пары вопрос-ответ"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO qa_pairs (question, answer, category, program_id, keywords)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                question,
                answer,
                category,
                program_id,
                json.dumps(keywords or [], ensure_ascii=False)
            ))
            
            conn.commit()
    
    def get_all_qa_pairs(self) -> List[Dict]:
        """Получение всех пар вопрос-ответ"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT qa.*, p.name as program_name 
                FROM qa_pairs qa 
                LEFT JOIN programs p ON qa.program_id = p.id
            ''')
            rows = cursor.fetchall()
            
            qa_pairs = []
            for row in rows:
                qa = dict(row)
                qa['keywords'] = json.loads(qa['keywords'])
                qa_pairs.append(qa)
            
            return qa_pairs
    
    def insert_user(self, telegram_id: int, username: str = None, 
                   first_name: str = None) -> int:
        """Добавление или обновление пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users (telegram_id, username, first_name)
                VALUES (?, ?, ?)
            ''', (telegram_id, username, first_name))
            
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
    
    def update_user_preferences(self, telegram_id: int, background: Dict = None, 
                              interests: List[str] = None, preferred_program: str = None):
        """Обновление предпочтений пользователя"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if background:
                updates.append('background = ?')
                params.append(json.dumps(background, ensure_ascii=False))
            
            if interests:
                updates.append('interests = ?')
                params.append(json.dumps(interests, ensure_ascii=False))
            
            if preferred_program:
                updates.append('preferred_program = ?')
                params.append(preferred_program)
            
            updates.append('updated_at = CURRENT_TIMESTAMP')
            params.append(telegram_id)
            
            query = f'UPDATE users SET {", ".join(updates)} WHERE telegram_id = ?'
            
            cursor.execute(query, params)
            conn.commit()
    
    def get_user_by_telegram_id(self, telegram_id: int) -> Optional[Dict]:
        """Получение пользователя по Telegram ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE telegram_id = ?', (telegram_id,))
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                if user['background']:
                    user['background'] = json.loads(user['background'])
                if user['interests']:
                    user['interests'] = json.loads(user['interests'])
                return user
            return None
    
    def save_course_recommendation(self, user_id: int, course_id: int, 
                                 score: float, reason: str):
        """Сохранение рекомендации курса"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO course_recommendations 
                (user_id, course_id, score, reason)
                VALUES (?, ?, ?, ?)
            ''', (user_id, course_id, score, reason))
            
            conn.commit()
    
    def populate_sample_qa_data(self):
        """Заполнение базовыми вопросами и ответами"""
        sample_qa = [
            {
                "question": "Какая продолжительность обучения в магистратуре?",
                "answer": "Обучение в магистратуре по программам 'Искусственный интеллект' и 'AI-продукты' длится 2 года (4 семестра).",
                "category": "general",
                "keywords": ["продолжительность", "срок", "длительность", "2 года"]
            },
            {
                "question": "Какие требования для поступления?",
                "answer": "Для поступления необходим диплом бакалавра или специалиста, прохождение вступительных испытаний по профилю программы. Рекомендуется наличие портфолио проектов.",
                "category": "admission",
                "keywords": ["поступление", "требования", "диплом", "бакалавр"]
            },
            {
                "question": "В чем разница между программами 'Искусственный интеллект' и 'AI-продукты'?",
                "answer": "Программа 'Искусственный интеллект' фокусируется на фундаментальных аспектах ИИ, алгоритмах и исследованиях. Программа 'AI-продукты' направлена на создание коммерческих AI-решений и продуктов.",
                "category": "programs",
                "keywords": ["разница", "отличие", "программы", "AI-продукты"]
            },
            {
                "question": "Какие карьерные возможности после окончания?",
                "answer": "Выпускники могут работать ML-инженерами, Data Scientists, AI-разработчиками, исследователями в области ИИ, продуктовыми менеджерами AI-продуктов.",
                "category": "career",
                "keywords": ["карьера", "работа", "трудоустройство", "ML-инженер", "Data Scientist"]
            }
        ]
        
        for qa in sample_qa:
            self.insert_qa_pair(
                qa["question"],
                qa["answer"],
                qa["category"],
                keywords=qa["keywords"]
            )
        
        logger.info("Базовые вопросы и ответы добавлены в базу данных")

if __name__ == "__main__":
    db = DatabaseManager()
    db.populate_sample_qa_data()
    print("База данных инициализирована и заполнена примерами") 