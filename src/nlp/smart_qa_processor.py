import logging
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.database.db_manager import DatabaseManager
from src.nlp.qa_processor import QAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartQAProcessor:
    """Улучшенная система Q&A - 'умный режим' без внешних API"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.qa_processor = QAProcessor(self.db)
        
        # Шаблоны для улучшенной обработки вопросов
        self.question_patterns = {
            'courses_by_topic': [
                r'курс.*?(machine learning|машинн\w+ обучени|ml)',
                r'курс.*?(deep learning|глубок\w+ обучени|нейронн\w+ сет)',
                r'курс.*?(computer vision|компьютерн\w+ зрени|cv|изображени)',
                r'курс.*?(nlp|natural language|обработк\w+ язык)',
                r'курс.*?(python|питон|программировани)',
                r'курс.*?(data science|данн\w+|аналитик)',
                r'курс.*?(статистик|математик)',
                r'курс.*?(алгоритм|структур\w+ данн)'
            ],
            'program_comparison': [
                r'разниц\w+ между программ',
                r'сравни\w*.*?программ',
                r'чем отличаются программы',
                r'какую программу выбрать',
                r'искусственный интеллект.*?ai[- ]?продукт',
                r'ai[- ]?продукт.*?искусственный интеллект'
            ],
            'learning_tracks': [
                r'траектори\w+',
                r'специализаци\w+',
                r'направлени\w+.*?обучени',
                r'варианты.*?обучени',
                r'пути.*?развити',
                r'какие.*?области.*?изучают',
                r'выборные.*?дисциплины',
                r'элективы',
                r'track'
            ],
            'admission_info': [
                r'как поступить',
                r'требовани\w+ для поступлени',
                r'вступительн\w+ испытани',
                r'экзамен',
                r'поступлени'
            ],
            'career_prospects': [
                r'карьерн\w+ перспектив',
                r'где работать',
                r'трудоустройств',
                r'работ\w+ после',
                r'зарплат'
            ],
            'duration_info': [
                r'сколько.*?длится',
                r'продолжительность',
                r'срок обучени',
                r'как долго'
            ]
        }
        
        # Умные ответы на основе анализа базы знаний
        self.smart_responses = {}
        self._build_smart_responses()
    
    def _build_smart_responses(self):
        """Построение умных ответов на основе данных из БД"""
        
        # Получаем все данные
        programs = self.db.get_all_programs()
        courses = self.db.get_all_courses()
        
        # Анализируем курсы по темам
        courses_by_topic = defaultdict(list)
        for course in courses:
            tags = course.get('tags', [])
            name = course['name'].lower()
            
            # Классифицируем курсы
            if any(tag.lower() in ['machine learning', 'ml'] for tag in tags) or 'машинное обучение' in name:
                courses_by_topic['machine_learning'].append(course)
            
            if any(tag.lower() in ['deep learning', 'neural networks'] for tag in tags) or 'глубокое обучение' in name:
                courses_by_topic['deep_learning'].append(course)
            
            if any(tag.lower() in ['computer vision', 'cv'] for tag in tags) or any(word in name for word in ['изображени', 'зрени', 'vision']):
                courses_by_topic['computer_vision'].append(course)
            
            if any(tag.lower() in ['nlp', 'natural language processing'] for tag in tags) or 'язык' in name:
                courses_by_topic['nlp'].append(course)
            
            if any(tag.lower() in ['python', 'programming'] for tag in tags) or 'python' in name:
                courses_by_topic['python'].append(course)
        
        # Строим умные ответы
        self.smart_responses['courses_by_topic'] = courses_by_topic
        
        # Сравнение программ
        if len(programs) >= 2:
            ai_program = next((p for p in programs if 'искусственный интеллект' in p['name'].lower()), None)
            ai_product_program = next((p for p in programs if 'ai-продукт' in p['name'].lower()), None)
            
            if ai_program and ai_product_program:
                ai_courses = self.db.get_courses_by_program(ai_program['id'])
                product_courses = self.db.get_courses_by_program(ai_product_program['id'])
                
                comparison = self._generate_program_comparison(ai_program, ai_product_program, ai_courses, product_courses)
                self.smart_responses['program_comparison'] = comparison
    
    def _generate_program_comparison(self, ai_program, product_program, ai_courses, product_courses):
        """Генерация сравнения программ на основе анализа курсов"""
        
        ai_tags = defaultdict(int)
        product_tags = defaultdict(int)
        
        # Анализируем теги курсов
        for course in ai_courses:
            for tag in course.get('tags', []):
                ai_tags[tag] += 1
        
        for course in product_courses:
            for tag in course.get('tags', []):
                product_tags[tag] += 1
        
        # Находим уникальные особенности
        ai_unique = []
        product_unique = []
        
        for tag, count in ai_tags.items():
            if count > product_tags.get(tag, 0):
                ai_unique.append(tag)
        
        for tag, count in product_tags.items():
            if count > ai_tags.get(tag, 0):
                product_unique.append(tag)
        
        comparison = {
            'ai_program': {
                'name': ai_program['name'],
                'courses_count': len(ai_courses),
                'unique_focus': ai_unique[:5],
                'description': ai_program.get('description', '')
            },
            'product_program': {
                'name': product_program['name'],
                'courses_count': len(product_courses),
                'unique_focus': product_unique[:5],
                'description': product_program.get('description', '')
            }
        }
        
        return comparison
    
    def detect_question_type(self, question: str) -> str:
        """Определение типа вопроса для умного ответа"""
        question_lower = question.lower()
        
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return question_type
        
        return 'general'
    
    def generate_smart_answer(self, user_question: str, user_context: Dict = None) -> Dict:
        """Генерация умного ответа без внешних API"""
        
        question_type = self.detect_question_type(user_question)
        
        # Сначала пробуем базовый Q&A
        base_result = self.qa_processor.get_answer(user_question)
        
        # Если базовый ответ хороший, используем его с улучшениями
        if base_result['confidence'] > 0.7:
            enhanced_answer = self._enhance_answer(base_result['answer'], question_type, user_question)
            return {
                'answer': enhanced_answer,
                'confidence': min(base_result['confidence'] + 0.1, 1.0),
                'method': 'smart_enhanced',
                'question_type': question_type
            }
        
        # Иначе генерируем умный ответ
        smart_answer = self._generate_smart_response(question_type, user_question, user_context)
        
        if smart_answer:
            return {
                'answer': smart_answer,
                'confidence': 0.85,
                'method': 'smart_local',
                'question_type': question_type,
                'is_ai_generated': True
            }
        
        # Fallback к базовому ответу
        return base_result
    
    def _enhance_answer(self, base_answer: str, question_type: str, user_question: str) -> str:
        """Улучшение базового ответа дополнительной информацией"""
        
        enhanced = base_answer
        
        # Добавляем связанные курсы если вопрос о курсах
        if question_type == 'courses_by_topic':
            topic = self._extract_topic_from_question(user_question)
            if topic and topic in self.smart_responses['courses_by_topic']:
                courses = self.smart_responses['courses_by_topic'][topic][:3]
                if courses:
                    enhanced += f"\n\n🎓 Релевантные курсы:\n"
                    for course in courses:
                        enhanced += f"• {course['name']} ({course.get('program_name', 'Неизвестно')})\n"
        
        # Добавляем связанные вопросы
        related = self.qa_processor.get_related_questions(user_question, top_k=2)
        if related:
            enhanced += f"\n\n💡 Возможно, вас также интересует:\n"
            for rel in related:
                enhanced += f"• {rel['question']}\n"
        
        enhanced += f"\n\n🤖 Ответ улучшен умной системой анализа"
        
        return enhanced
    
    def _generate_smart_response(self, question_type: str, user_question: str, user_context: Dict = None) -> Optional[str]:
        """Генерация умного ответа на основе анализа базы знаний"""
        
        if question_type == 'courses_by_topic':
            return self._answer_courses_by_topic(user_question)
        
        elif question_type == 'program_comparison':
            return self._answer_program_comparison()
        
        elif question_type == 'learning_tracks':
            return self._answer_learning_tracks()
        
        elif question_type == 'admission_info':
            return self._answer_admission_info()
        
        elif question_type == 'career_prospects':
            return self._answer_career_prospects()
        
        elif question_type == 'duration_info':
            return self._answer_duration_info()
        
        return None
    
    def _extract_topic_from_question(self, question: str) -> Optional[str]:
        """Извлечение темы из вопроса о курсах"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['machine learning', 'машинное обучение', 'мо']):
            return 'machine_learning'
        elif any(word in question_lower for word in ['deep learning', 'глубокое обучение', 'нейронные сети']):
            return 'deep_learning'
        elif any(word in question_lower for word in ['computer vision', 'компьютерное зрение', 'cv', 'изображение']):
            return 'computer_vision'
        elif any(word in question_lower for word in ['nlp', 'natural language', 'язык']):
            return 'nlp'
        elif 'python' in question_lower:
            return 'python'
        
        return None
    
    def _answer_courses_by_topic(self, question: str) -> str:
        """Ответ на вопросы о курсах по теме"""
        topic = self._extract_topic_from_question(question)
        
        if not topic or topic not in self.smart_responses['courses_by_topic']:
            return "🔍 Не могу найти курсы по указанной теме. Попробуйте переформулировать вопрос."
        
        courses = self.smart_responses['courses_by_topic'][topic]
        
        if not courses:
            return "📚 К сожалению, курсы по этой теме не найдены в базе данных."
        
        topic_names = {
            'machine_learning': 'машинному обучению',
            'deep_learning': 'глубокому обучению',
            'computer_vision': 'компьютерному зрению',
            'nlp': 'обработке естественного языка',
            'python': 'Python'
        }
        
        answer = f"🎓 Курсы по {topic_names.get(topic, 'указанной теме')}:\n\n"
        
        for i, course in enumerate(courses[:5], 1):
            answer += f"{i}. {course['name']}\n"
            answer += f"   📚 Программа: {course.get('program_name', 'Неизвестно')}\n"
            answer += f"   📅 Семестр: {course.get('semester', 'Не указан')}\n"
            if course.get('tags'):
                answer += f"   🏷️ Теги: {', '.join(course['tags'][:3])}\n"
            answer += "\n"
        
        answer += "💡 Для получения персональных рекомендаций нажмите 'Получить рекомендации'"
        
        return answer
    
    def _answer_program_comparison(self) -> str:
        """Ответ на вопросы о сравнении программ"""
        if 'program_comparison' not in self.smart_responses:
            return "📊 Информация о сравнении программ временно недоступна."
        
        comp = self.smart_responses['program_comparison']
        ai_prog = comp['ai_program']
        prod_prog = comp['product_program']
        
        answer = "🎓 Сравнение программ ИТМО:\n\n"
        
        answer += f"🧠 **{ai_prog['name']}**\n"
        answer += f"📚 Курсов: {ai_prog['courses_count']}\n"
        if ai_prog['unique_focus']:
            answer += f"🎯 Особый фокус: {', '.join(ai_prog['unique_focus'][:3])}\n"
        answer += f"📖 {ai_prog['description'][:200]}...\n\n"
        
        answer += f"🚀 **{prod_prog['name']}**\n"
        answer += f"📚 Курсов: {prod_prog['courses_count']}\n"
        if prod_prog['unique_focus']:
            answer += f"🎯 Особый фокус: {', '.join(prod_prog['unique_focus'][:3])}\n"
        answer += f"📖 {prod_prog['description'][:200]}...\n\n"
        
        answer += "💡 **Рекомендации:**\n"
        answer += "• Выбирайте 'Искусственный интеллект' для углубленного изучения ИИ\n"
        answer += "• Выбирайте 'AI-продукты' для практического применения ИИ в продуктах\n\n"
        
        answer += "🤖 Анализ на основе сравнения учебных планов"
        
        return answer
    
    def _answer_learning_tracks(self) -> str:
        """Ответ на вопросы о траекториях и специализациях"""
        
        # Получаем курсы по программам
        programs = self.db.get_all_programs()
        
        answer = "🛤️ Траектории обучения в ИТМО:\n\n"
        
        for program in programs:
            courses = self.db.get_courses_by_program(program['id'])
            
            # Анализируем курсы по категориям
            mandatory_courses = [c for c in courses if c.get('is_mandatory', False)]
            elective_courses = [c for c in courses if not c.get('is_mandatory', True)]
            
            # Группируем по семестрам
            courses_by_semester = {}
            for course in courses:
                semester = course.get('semester', 'Не указан')
                if semester not in courses_by_semester:
                    courses_by_semester[semester] = []
                courses_by_semester[semester].append(course)
            
            answer += f"📚 **{program['name']}**\n"
            answer += f"📖 Всего курсов: {len(courses)}\n"
            answer += f"⭐ Обязательных: {len(mandatory_courses)}\n"
            answer += f"🎯 Элективных: {len(elective_courses)}\n\n"
            
            # Показываем основные направления
            if 'courses_by_topic' in self.smart_responses:
                topics = self.smart_responses['courses_by_topic']
                program_topics = []
                
                for topic, topic_courses in topics.items():
                    # Фильтруем курсы этой программы
                    program_topic_courses = [c for c in topic_courses if c.get('program') == program['id']]
                    if program_topic_courses:
                        topic_names = {
                            'machine_learning': 'Машинное обучение',
                            'deep_learning': 'Глубокое обучение', 
                            'computer_vision': 'Компьютерное зрение',
                            'nlp': 'Обработка языка',
                            'python': 'Python разработка'
                        }
                        program_topics.append(f"  • {topic_names.get(topic, topic.title())} ({len(program_topic_courses)} курсов)")
                
                if program_topics:
                    answer += "🎯 **Основные направления:**\n"
                    answer += "\n".join(program_topics[:5])
                    answer += "\n\n"
            
            # Показываем структуру по семестрам
            answer += "📅 **Структура обучения:**\n"
            sorted_semesters = sorted([k for k in courses_by_semester.keys() if k != 'Не указан'])
            for semester in sorted_semesters[:4]:  # Показываем первые 4 семестра
                sem_courses = courses_by_semester[semester]
                answer += f"  {semester}: {len(sem_courses)} курсов\n"
            
            answer += "\n" + "─" * 30 + "\n\n"
        
        # Общие рекомендации
        answer += "💡 **Варианты специализации:**\n\n"
        answer += "🤖 **Для исследователей:**\n"
        answer += "• Фокус на теоретические курсы\n"
        answer += "• Машинное и глубокое обучение\n"
        answer += "• Математическая статистика\n\n"
        
        answer += "💼 **Для практиков:**\n"
        answer += "• Python разработка\n"
        answer += "• Веб-приложения и продукты\n"
        answer += "• Прикладные проекты\n\n"
        
        answer += "🎯 **Для специалистов по данным:**\n"
        answer += "• Анализ данных\n"
        answer += "• Computer Vision или NLP\n"
        answer += "• Рекомендательные системы\n\n"
        
        answer += "📋 **Как выбрать:**\n"
        answer += "1. Определите цель: исследования или практика\n"
        answer += "2. Выберите основную программу\n"
        answer += "3. Сформируйте портфель элективов\n"
        answer += "4. Консультируйтесь с кураторами\n\n"
        
        answer += "🔍 Для персональных рекомендаций используйте 'Получить рекомендации'"
        
        return answer
    
    def _answer_admission_info(self) -> str:
        """Ответ на вопросы о поступлении"""
        return """🎯 Требования для поступления:

📋 **Необходимые документы:**
• Диплом бакалавра или специалиста
• Транскрипт с оценками
• Мотивационное письмо
• Портфолио проектов (рекомендуется)

💻 **Вступительные испытания:**
• Собеседование по профилю программы
• Техническое задание (возможно)
• Английский язык (базовый уровень)

📅 **Сроки подачи документов:**
• Обычно до июля текущего года
• Точные даты смотрите на сайте ИТМО

🔗 **Подробнее:** https://abit.itmo.ru/

💡 Рекомендуется иметь опыт в программировании и математике"""
    
    def _answer_career_prospects(self) -> str:
        """Ответ на вопросы о карьерных перспективах"""
        return """💼 Карьерные перспективы выпускников:

🤖 **После "Искусственный интеллект":**
• Data Scientist / ML Engineer
• Research Scientist
• AI Architect
• Computer Vision Engineer
• NLP Engineer

🚀 **После "AI-продукты":**
• Product Manager (AI/ML)
• AI Product Owner
• ML Solutions Architect
• Technical Product Manager
• AI Consultant

💰 **Средние зарплаты в РФ:**
• Junior: 80-150k руб/мес
• Middle: 150-300k руб/мес  
• Senior: 300-500k+ руб/мес

🌍 **Международные возможности:**
• Удаленная работа в зарубежных компаниях
• Релокация в IT-хабы
• Участие в международных проектах

🎓 Диплом ИТМО высоко ценится в IT-индустрии"""
    
    def _answer_duration_info(self) -> str:
        """Ответ на вопросы о продолжительности обучения"""
        return """⏱️ Продолжительность обучения:

📅 **Стандартная программа:**
• 2 года (4 семестра)
• Очная форма обучения
• Полная занятость

📚 **Структура:**
• 1-й год: Базовые курсы + специализация
• 2-й год: Продвинутые темы + дипломная работа

⚡ **Интенсивность:**
• ~20-30 часов в неделю
• Лекции, семинары, практические работы
• Самостоятельная работа и проекты

🎓 **Выпуск:**
• Защита магистерской диссертации
• Получение диплома магистра ИТМО

💡 Возможны индивидуальные траектории обучения"""

if __name__ == "__main__":
    # Тестирование умной системы
    smart_qa = SmartQAProcessor()
    
    print("🧠 Тестирование умной системы Q&A")
    print("=" * 50)
    
    test_questions = [
        "Какие курсы по машинному обучению есть?",
        "В чем разница между программами?",
        "Как поступить в магистратуру?",
        "Сколько длится обучение?",
        "Где можно работать после выпуска?"
    ]
    
    for question in test_questions:
        print(f"\n❓ {question}")
        result = smart_qa.generate_smart_answer(question)
        print(f"🤖 Тип: {result.get('question_type', 'general')}")
        print(f"📊 Метод: {result['method']}")
        print(f"💡 Ответ: {result['answer'][:150]}...")
        print("-" * 30) 