import logging
from typing import List, Dict, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseRecommender:
    def __init__(self, db_manager: DatabaseManager = None):
        """Инициализация системы рекомендаций курсов"""
        self.db = db_manager or DatabaseManager()
        
        # Словарь ключевых слов для извлечения интересов
        self.interest_keywords = {
            'machine_learning': [
                'машинное обучение', 'machine learning', 'ml', 'алгоритмы машинного обучения',
                'классификация', 'регрессия', 'кластеризация', 'обучение с учителем',
                'обучение без учителя', 'supervised learning', 'unsupervised learning'
            ],
            'deep_learning': [
                'глубокое обучение', 'deep learning', 'нейронные сети', 'neural networks',
                'cnn', 'rnn', 'lstm', 'transformer', 'свёрточные сети', 'рекуррентные сети'
            ],
            'computer_vision': [
                'компьютерное зрение', 'computer vision', 'cv', 'обработка изображений',
                'распознавание образов', 'image processing', 'opencv', 'детекция объектов'
            ],
            'nlp': [
                'обработка естественного языка', 'natural language processing', 'nlp',
                'анализ текста', 'text mining', 'sentiment analysis', 'чат-боты', 'bert'
            ],
            'data_science': [
                'data science', 'анализ данных', 'большие данные', 'big data',
                'статистика', 'analytics', 'pandas', 'numpy', 'визуализация данных'
            ],
            'python': [
                'python', 'программирование на python', 'django', 'flask', 'fastapi',
                'pandas', 'numpy', 'scikit-learn', 'pytorch', 'tensorflow'
            ],
            'research': [
                'исследования', 'research', 'научная работа', 'публикации', 'статьи',
                'эксперименты', 'analysis', 'методология'
            ],
            'product': [
                'продукт', 'product', 'продакт-менеджмент', 'product management',
                'бизнес', 'стартап', 'коммерциализация', 'метрики', 'a/b тестирование'
            ],
            'robotics': [
                'робототехника', 'robotics', 'роботы', 'автоматизация', 'sensors',
                'управление', 'киберфизические системы'
            ],
            'math': [
                'математика', 'mathematics', 'статистика', 'probability', 'вероятность',
                'линейная алгебра', 'linear algebra', 'оптимизация', 'optimization'
            ]
        }
        
        # Приоритеты тегов для разных программ
        self.program_priorities = {
            'AI': {
                'machine_learning': 1.0,
                'deep_learning': 1.0,
                'research': 0.9,
                'math': 0.8,
                'python': 0.7,
                'computer_vision': 0.8,
                'nlp': 0.8
            },
            'AI_Product': {
                'product': 1.0,
                'machine_learning': 0.9,
                'data_science': 0.9,
                'python': 0.8,
                'deep_learning': 0.7,
                'research': 0.6
            }
        }
    
    def extract_interests_from_text(self, text: str) -> Dict[str, float]:
        """Извлечение интересов из текста пользователя"""
        if not text:
            return {}
        
        text_lower = text.lower()
        interests = {}
        
        for interest, keywords in self.interest_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    # Более длинные фразы получают больший вес
                    weight = len(keyword.split()) * 0.3 + 0.7
                    score += weight
            
            if score > 0:
                interests[interest] = min(score, 1.0)  # Нормализуем до 1.0
        
        return interests
    
    def calculate_course_score(self, course: Dict, user_interests: Dict[str, float], 
                             preferred_program: str = None) -> Tuple[float, str]:
        """Расчет релевантности курса для пользователя"""
        if not user_interests:
            return 0.0, "Нет информации об интересах пользователя"
        
        # Базовый скор
        base_score = 0.0
        reasons = []
        
        # Анализируем теги курса
        course_tags = course.get('tags', [])
        course_name = course.get('name', '').lower()
        course_description = course.get('description', '').lower()
        
        # Объединяем все текстовые данные курса
        course_text = f"{course_name} {course_description} {' '.join(course_tags)}".lower()
        
        # Создаем расширенное сопоставление интересов с тегами
        interest_to_tags_mapping = {
            'computer_vision': ['Computer Vision', 'CV', 'Vision'],
            'machine_learning': ['Machine Learning', 'ML'],
            'deep_learning': ['Deep Learning', 'Neural Networks'],
            'nlp': ['NLP', 'Natural Language Processing'],
            'data_science': ['Data Science', 'Analytics', 'Statistics', 'Math'],
            'python': ['Python', 'Programming'],
            'research': ['Research', 'Analysis'],
            'algorithms': ['Algorithms', 'Programming'],
            'math': ['Math', 'Statistics'],
            'web_development': ['Web Development', 'Programming']
        }
        
        # Проверяем совпадения с интересами пользователя
        for interest, user_score in user_interests.items():
            interest_score = 0.0
            
            # Получаем возможные теги для данного интереса
            possible_tags = interest_to_tags_mapping.get(interest, [interest.title()])
            
            # Проверяем прямые совпадения в названии курса
            interest_keywords = self.interest_keywords.get(interest, [])
            for keyword in interest_keywords:
                if keyword in course_name:
                    interest_score += 0.5
                    reasons.append(f"Совпадение '{keyword}' в названии курса")
            
            # Проверяем совпадения в тегах курса
            for tag in course_tags:
                for possible_tag in possible_tags:
                    if possible_tag.lower() in tag.lower() or tag.lower() in possible_tag.lower():
                        interest_score += 0.7
                        reasons.append(f"Совпадение по тегу '{tag}'")
            
            # Дополнительная проверка для Computer Vision
            if interest == 'computer_vision':
                cv_keywords = ['изображен', 'vision', 'зрение', 'обработка изображений', 'генерация изображений']
                for keyword in cv_keywords:
                    if keyword in course_name:
                        interest_score += 0.8
                        reasons.append(f"CV ключевое слово '{keyword}'")
            
            # Применяем приоритеты программы
            program_weight = 1.0
            if preferred_program and preferred_program in self.program_priorities:
                program_weight = self.program_priorities[preferred_program].get(interest, 0.7)
            
            # Добавляем к общему скору
            if interest_score > 0:
                base_score += min(interest_score, 1.0) * user_score * program_weight
        
        # Бонус за обязательные курсы только если нет других совпадений
        if base_score == 0 and course.get('is_mandatory', False):
            base_score += 0.2
            reasons.append("Обязательный курс программы")
        
        # Бонус за соответствие предпочитаемой программе
        if preferred_program and course.get('program') == preferred_program:
            if base_score > 0:  # Только если есть содержательные совпадения
                base_score += 0.2
                reasons.append(f"Курс из предпочитаемой программы")
        
        # Нормализуем финальный скор
        final_score = min(base_score, 1.0)
        
        reason_text = "; ".join(reasons[:3]) if reasons else "Общие рекомендации"
        
        return final_score, reason_text
    
    def recommend_courses(self, user_interests: Dict[str, float], 
                         preferred_program: str = None, 
                         top_k: int = 5,
                         min_score: float = 0.1) -> List[Dict]:
        """Рекомендация курсов для пользователя"""
        
        # Получаем все курсы
        all_courses = self.db.get_all_courses()
        
        if not all_courses:
            logger.warning("Нет курсов в базе данных")
            return []
        
        recommendations = []
        
        for course in all_courses:
            score, reason = self.calculate_course_score(
                course, user_interests, preferred_program
            )
            
            if score >= min_score:
                recommendations.append({
                    'course': course,
                    'score': score,
                    'reason': reason,
                    'course_id': course['id'],
                    'course_name': course['name'],
                    'program_name': course.get('program_name', 'Unknown'),
                    'credits': course.get('credits', 0),
                    'semester': course.get('semester', 'Unknown'),
                    'is_mandatory': course.get('is_mandatory', False)
                })
        
        # Сортируем по скору
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Возвращаем топ-k рекомендаций
        return recommendations[:top_k]
    
    def recommend_courses_from_text(self, user_text: str, 
                                  preferred_program: str = None,
                                  top_k: int = 5) -> List[Dict]:
        """Рекомендация курсов на основе текста пользователя"""
        
        # Извлекаем интересы из текста
        interests = self.extract_interests_from_text(user_text)
        
        if not interests:
            logger.info("Не удалось извлечь интересы из текста")
            # Возвращаем общие рекомендации
            return self.get_general_recommendations(preferred_program, top_k)
        
        logger.info(f"Извлеченные интересы: {interests}")
        
        return self.recommend_courses(interests, preferred_program, top_k)
    
    def get_general_recommendations(self, preferred_program: str = None, 
                                  top_k: int = 5) -> List[Dict]:
        """Общие рекомендации курсов"""
        all_courses = self.db.get_all_courses()
        
        if not all_courses:
            return []
        
        # Фильтруем по программе если указана
        if preferred_program:
            filtered_courses = [c for c in all_courses if c.get('program') == preferred_program]
        else:
            filtered_courses = all_courses
        
        # Сортируем: сначала обязательные, потом по алфавиту
        filtered_courses.sort(key=lambda x: (not x.get('is_mandatory', False), x.get('name', '')))
        
        recommendations = []
        for course in filtered_courses[:top_k]:
            recommendations.append({
                'course': course,
                'score': 0.5,  # Средний скор для общих рекомендаций
                'reason': 'Общая рекомендация программы',
                'course_id': course['id'],
                'course_name': course['name'],
                'program_name': course.get('program_name', 'Unknown'),
                'credits': course.get('credits', 0),
                'semester': course.get('semester', 'Unknown'),
                'is_mandatory': course.get('is_mandatory', False)
            })
        
        return recommendations
    
    def save_recommendations(self, telegram_id: int, recommendations: List[Dict]):
        """Сохранение рекомендаций в базу данных"""
        user = self.db.get_user_by_telegram_id(telegram_id)
        if not user:
            logger.error(f"Пользователь с telegram_id {telegram_id} не найден")
            return
        
        for rec in recommendations:
            self.db.save_course_recommendation(
                user['id'],
                rec['course_id'],
                rec['score'],
                rec['reason']
            )
    
    def get_interest_suggestions(self) -> List[str]:
        """Получение списка возможных интересов для пользователя"""
        suggestions = []
        for interest, keywords in self.interest_keywords.items():
            # Берем первое ключевое слово как основное название интереса
            main_keyword = keywords[0].title()
            suggestions.append(main_keyword)
        
        return sorted(suggestions)
    
    def analyze_user_background(self, background_text: str) -> Dict:
        """Анализ бэкграунда пользователя"""
        analysis = {
            'experience_level': 'beginner',
            'technical_skills': [],
            'domains': [],
            'interests': self.extract_interests_from_text(background_text)
        }
        
        text_lower = background_text.lower()
        
        # Определяем уровень опыта
        if any(word in text_lower for word in ['опыт', 'работал', 'работаю', 'experience', 'senior', 'lead']):
            analysis['experience_level'] = 'experienced'
        elif any(word in text_lower for word in ['изучал', 'изучаю', 'начинающий', 'beginner', 'junior']):
            analysis['experience_level'] = 'intermediate'
        
        # Технические навыки
        tech_skills = {
            'Python': ['python', 'питон'],
            'Java': ['java'],
            'C++': ['c++', 'cpp'],
            'JavaScript': ['javascript', 'js', 'node.js'],
            'SQL': ['sql', 'database', 'база данных'],
            'Git': ['git', 'github', 'gitlab']
        }
        
        for skill, keywords in tech_skills.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis['technical_skills'].append(skill)
        
        return analysis

# Пример использования
if __name__ == "__main__":
    recommender = CourseRecommender()
    
    # Тестовый текст пользователя
    test_text = """
    У меня есть опыт программирования на Python, работал с pandas и numpy.
    Интересуюсь машинным обучением и хочу изучить глубокое обучение.
    Также интересует computer vision и обработка изображений.
    """
    
    print("Тестирование системы рекомендаций:")
    print("=" * 50)
    print(f"Текст пользователя: {test_text}")
    print()
    
    # Извлекаем интересы
    interests = recommender.extract_interests_from_text(test_text)
    print(f"Извлеченные интересы: {interests}")
    print()
    
    # Получаем рекомендации
    recommendations = recommender.recommend_courses_from_text(test_text, preferred_program='AI')
    
    print("Рекомендованные курсы:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['course_name']} (Программа: {rec['program_name']})")
        print(f"   Скор: {rec['score']:.2f}")
        print(f"   Обоснование: {rec['reason']}")
        print(f"   Семестр: {rec['semester']}, Кредиты: {rec['credits']}")
        print() 