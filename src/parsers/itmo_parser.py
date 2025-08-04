import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

from config import (
    ITMO_AI_URL, 
    ITMO_AI_PRODUCT_URL, 
    REQUEST_TIMEOUT, 
    DELAY_BETWEEN_REQUESTS,
    JSON_DATA_PATH
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Course:
    name: str
    description: str
    credits: int
    semester: str
    is_mandatory: bool
    program: str  # "AI" или "AI_Product"
    tags: List[str]
    prerequisites: List[str]

@dataclass
class Program:
    name: str
    description: str
    duration: str
    courses: List[Course]
    admission_requirements: List[str]
    career_prospects: List[str]

class ITMOParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Получение содержимого страницы"""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Ошибка при получении страницы {url}: {e}")
            return None
    
    def parse_ai_program(self) -> Optional[Program]:
        """Парсинг программы 'Искусственный интеллект'"""
        logger.info("Парсинг программы 'Искусственный интеллект'...")
        soup = self.get_page_content(ITMO_AI_URL)
        if not soup:
            return None
            
        return self._extract_program_data(soup, "AI", "Искусственный интеллект")
    
    def parse_ai_product_program(self) -> Optional[Program]:
        """Парсинг программы 'AI-продукты'"""
        logger.info("Парсинг программы 'AI-продукты'...")
        soup = self.get_page_content(ITMO_AI_PRODUCT_URL)
        if not soup:
            return None
            
        return self._extract_program_data(soup, "AI_Product", "AI-продукты")
    
    def _extract_program_data(self, soup: BeautifulSoup, program_id: str, program_name: str) -> Program:
        """Извлечение данных программы из HTML"""
        
        # Извлечение описания программы
        description = self._extract_description(soup)
        
        # Извлечение продолжительности обучения
        duration = self._extract_duration(soup)
        
        # Извлечение требований к поступлению
        admission_requirements = self._extract_admission_requirements(soup)
        
        # Извлечение карьерных перспектив
        career_prospects = self._extract_career_prospects(soup)
        
        # Извлечение курсов
        courses = self._extract_courses(soup, program_id)
        
        return Program(
            name=program_name,
            description=description,
            duration=duration,
            courses=courses,
            admission_requirements=admission_requirements,
            career_prospects=career_prospects
        )
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Извлечение описания программы"""
        # Ищем описание в различных возможных селекторах
        selectors = [
            '.program-description',
            '.description',
            '.about-program',
            '.program-info p',
            'h1 + p',
            '.content p:first-of-type'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
                
        # Если не нашли специфичный селектор, ищем первый параграф с достаточным текстом
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 100:  # Минимальная длина для описания
                return text
                
        return "Описание программы не найдено"
    
    def _extract_duration(self, soup: BeautifulSoup) -> str:
        """Извлечение продолжительности обучения"""
        # Ищем информацию о продолжительности
        duration_keywords = ['срок', 'продолжительность', 'длительность', 'года', 'лет']
        
        text = soup.get_text().lower()
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in duration_keywords):
                # Ищем цифры в строке
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return f"{numbers[0]} года"
                    
        return "2 года"  # Значение по умолчанию для магистратуры
    
    def _extract_admission_requirements(self, soup: BeautifulSoup) -> List[str]:
        """Извлечение требований к поступлению"""
        requirements = []
        
        # Ищем секции с требованиями
        requirement_sections = soup.find_all(text=re.compile(r'(требования|поступление|admission)', re.I))
        
        for section in requirement_sections:
            parent = section.parent
            if parent:
                # Ищем списки рядом с найденным текстом
                lists = parent.find_next_siblings(['ul', 'ol']) or parent.find_all(['ul', 'ol'])
                for ul in lists:
                    items = ul.find_all('li')
                    for item in items:
                        req = item.get_text(strip=True)
                        if req and len(req) > 10:
                            requirements.append(req)
        
        # Если не нашли специфичные требования, добавляем общие
        if not requirements:
            requirements = [
                "Диплом бакалавра или специалиста",
                "Вступительные испытания по профилю программы",
                "Портфолио проектов (при наличии)"
            ]
            
        return requirements
    
    def _extract_career_prospects(self, soup: BeautifulSoup) -> List[str]:
        """Извлечение карьерных перспектив"""
        prospects = []
        
        # Ищем секции с карьерными перспективами
        career_keywords = ['карьера', 'работа', 'трудоустройство', 'профессия', 'career']
        text_content = soup.get_text().lower()
        
        for keyword in career_keywords:
            if keyword in text_content:
                # Находим секции, содержащие эти ключевые слова
                elements = soup.find_all(text=re.compile(keyword, re.I))
                for element in elements:
                    parent = element.parent
                    if parent:
                        lists = parent.find_next_siblings(['ul', 'ol']) or parent.find_all(['ul', 'ol'])
                        for ul in lists:
                            items = ul.find_all('li')
                            for item in items:
                                prospect = item.get_text(strip=True)
                                if prospect and len(prospect) > 5:
                                    prospects.append(prospect)
        
        # Если не нашли специфичные перспективы, добавляем общие для AI
        if not prospects:
            prospects = [
                "ML-инженер",
                "Data Scientist",
                "AI-разработчик",
                "Исследователь в области ИИ",
                "Продуктовый менеджер AI-продуктов"
            ]
            
        return list(set(prospects))  # Убираем дубликаты
    
    def _extract_courses(self, soup: BeautifulSoup, program_id: str) -> List[Course]:
        """Извлечение курсов из учебного плана"""
        courses = []
        
        # Ищем таблицы с учебным планом
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Пропускаем заголовок
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    course_name = cells[0].get_text(strip=True)
                    
                    # Пропускаем пустые строки и заголовки
                    if not course_name or len(course_name) < 3:
                        continue
                        
                    # Извлекаем кредиты
                    credits = self._extract_credits(cells)
                    
                    # Определяем семестр
                    semester = self._extract_semester(cells, row)
                    
                    # Определяем обязательность курса
                    is_mandatory = self._is_mandatory_course(course_name, cells)
                    
                    # Генерируем теги на основе названия
                    tags = self._generate_tags(course_name)
                    
                    course = Course(
                        name=course_name,
                        description=f"Курс по программе {program_id}",
                        credits=credits,
                        semester=semester,
                        is_mandatory=is_mandatory,
                        program=program_id,
                        tags=tags,
                        prerequisites=[]
                    )
                    
                    courses.append(course)
        
        # Если не нашли курсы в таблице, ищем в списках
        if not courses:
            courses = self._extract_courses_from_lists(soup, program_id)
            
        return courses
    
    def _extract_credits(self, cells) -> int:
        """Извлечение количества кредитов"""
        for cell in cells:
            text = cell.get_text(strip=True)
            # Ищем числа, которые могут быть кредитами
            numbers = re.findall(r'\b(\d+)\b', text)
            for num in numbers:
                if 1 <= int(num) <= 12:  # Разумный диапазон для кредитов
                    return int(num)
        return 3  # Значение по умолчанию
    
    def _extract_semester(self, cells, row) -> str:
        """Извлечение семестра"""
        for cell in cells:
            text = cell.get_text(strip=True).lower()
            if 'семестр' in text or 'semester' in text:
                numbers = re.findall(r'\d+', text)
                if numbers:
                    return f"{numbers[0]} семестр"
        
        # Пытаемся определить по позиции в таблице
        return "1 семестр"  # Значение по умолчанию
    
    def _is_mandatory_course(self, course_name: str, cells) -> bool:
        """Определение обязательности курса"""
        # Ключевые слова для обязательных курсов
        mandatory_keywords = ['обязательный', 'mandatory', 'required', 'базовый']
        optional_keywords = ['выборный', 'optional', 'elective', 'элективный']
        
        text = ' '.join([cell.get_text(strip=True).lower() for cell in cells])
        text += ' ' + course_name.lower()
        
        if any(keyword in text for keyword in optional_keywords):
            return False
        if any(keyword in text for keyword in mandatory_keywords):
            return True
            
        # По умолчанию считаем обязательным
        return True
    
    def _generate_tags(self, course_name: str) -> List[str]:
        """Генерация тегов на основе названия курса"""
        tags = []
        course_lower = course_name.lower()
        
        # Словарь ключевых слов и соответствующих тегов
        keyword_tags = {
            'машинное обучение': ['ML', 'Machine Learning'],
            'machine learning': ['ML', 'Machine Learning'],
            'глубокое обучение': ['Deep Learning', 'Neural Networks'],
            'deep learning': ['Deep Learning', 'Neural Networks'],
            'нейронные сети': ['Neural Networks', 'Deep Learning'],
            'neural networks': ['Neural Networks', 'Deep Learning'],
            'python': ['Python', 'Programming'],
            'программирование': ['Programming', 'Development'],
            'data': ['Data Science', 'Analytics'],
            'данные': ['Data Science', 'Analytics'],
            'алгоритм': ['Algorithms', 'CS'],
            'статистика': ['Statistics', 'Math'],
            'математика': ['Math', 'Statistics'],
            'computer vision': ['Computer Vision', 'CV'],
            'компьютерное зрение': ['Computer Vision', 'CV'],
            'nlp': ['NLP', 'Text Processing'],
            'обработка языка': ['NLP', 'Text Processing'],
            'reinforcement learning': ['RL', 'Reinforcement Learning'],
            'обучение с подкреплением': ['RL', 'Reinforcement Learning']
        }
        
        for keyword, keyword_tags_list in keyword_tags.items():
            if keyword in course_lower:
                tags.extend(keyword_tags_list)
        
        return list(set(tags))  # Убираем дубликаты
    
    def _extract_courses_from_lists(self, soup: BeautifulSoup, program_id: str) -> List[Course]:
        """Альтернативный метод извлечения курсов из списков"""
        courses = []
        
        # Ищем списки с курсами
        lists = soup.find_all(['ul', 'ol'])
        
        for ul in lists:
            items = ul.find_all('li')
            for item in items:
                course_name = item.get_text(strip=True)
                
                # Фильтруем по длине и содержанию
                if len(course_name) > 10 and not course_name.lower().startswith(('http', 'www', 'подробнее')):
                    course = Course(
                        name=course_name,
                        description=f"Курс по программе {program_id}",
                        credits=3,
                        semester="1 семестр",
                        is_mandatory=True,
                        program=program_id,
                        tags=self._generate_tags(course_name),
                        prerequisites=[]
                    )
                    courses.append(course)
        
        return courses
    
    def save_to_json(self, programs: List[Program], filename: str = "programs_data.json"):
        """Сохранение данных в JSON файл"""
        data = []
        for program in programs:
            program_data = {
                'name': program.name,
                'description': program.description,
                'duration': program.duration,
                'admission_requirements': program.admission_requirements,
                'career_prospects': program.career_prospects,
                'courses': []
            }
            
            for course in program.courses:
                course_data = {
                    'name': course.name,
                    'description': course.description,
                    'credits': course.credits,
                    'semester': course.semester,
                    'is_mandatory': course.is_mandatory,
                    'program': course.program,
                    'tags': course.tags,
                    'prerequisites': course.prerequisites
                }
                program_data['courses'].append(course_data)
            
            data.append(program_data)
        
        filepath = JSON_DATA_PATH + filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные сохранены в {filepath}")
    
    def parse_all_programs(self) -> List[Program]:
        """Парсинг всех программ"""
        programs = []
        
        # Парсинг программы AI
        ai_program = self.parse_ai_program()
        if ai_program:
            programs.append(ai_program)
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Парсинг программы AI Product
        ai_product_program = self.parse_ai_product_program()
        if ai_product_program:
            programs.append(ai_product_program)
        
        return programs

if __name__ == "__main__":
    parser = ITMOParser()
    programs = parser.parse_all_programs()
    
    if programs:
        parser.save_to_json(programs)
        print(f"Успешно спарсено {len(programs)} программ")
        for program in programs:
            print(f"- {program.name}: {len(program.courses)} курсов")
    else:
        print("Не удалось спарсить программы") 