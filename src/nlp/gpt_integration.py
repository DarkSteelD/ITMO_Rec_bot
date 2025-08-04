import openai
import os
import logging
from typing import Dict, List, Optional, Tuple
import json

from src.database.db_manager import DatabaseManager
from src.nlp.qa_processor import QAProcessor
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTIntegration:
    def __init__(self, db_manager: DatabaseManager = None):
        """Инициализация GPT интеграции с RAG"""
        self.db = db_manager or DatabaseManager()
        self.qa_processor = QAProcessor(self.db)
        
        # Настройки API
        self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('GPT_API_KEY')
        self.api_base = os.getenv('GPT_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('GPT_MODEL', 'gpt-3.5-turbo')
        
        # Альтернативные API (бесплатные/дешевые)
        self.use_free_api = os.getenv('USE_FREE_GPT', 'false').lower() == 'true'
        
        if self.api_key:
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            logger.info(f"GPT API инициализирован: {self.model}")
        else:
            logger.warning("GPT API ключ не найден. Используется только база знаний.")
        
        # Системный промпт для ИТМО бота
        self.system_prompt = """
Ты - специализированный помощник для абитуриентов магистерских программ ИТМО по искусственному интеллекту.

ТВОЯ РОЛЬ:
- Отвечаешь на вопросы о программах "Искусственный интеллект" и "AI-продукты" 
- Помогаешь выбирать курсы и строить учебный план
- Консультируешь по поступлению и карьерным перспективам
- Используешь ТОЛЬКО информацию из предоставленного контекста

ПРАВИЛА:
- Отвечай дружелюбно и профессионально
- Если информации нет в контексте, честно скажи об этом
- Рекомендуй конкретные курсы на основе интересов пользователя
- Используй эмодзи для лучшего восприятия
- Отвечай на русском языке

КОНТЕКСТ из базы знаний будет предоставлен отдельно.
        """.strip()
    
    def is_available(self) -> bool:
        """Проверка доступности GPT API"""
        return self.api_key is not None
    
    def get_relevant_context(self, user_question: str, max_items: int = 5) -> str:
        """Получение релевантного контекста из базы знаний"""
        context_parts = []
        
        # 1. Ищем похожие Q&A
        qa_result = self.qa_processor.get_answer(user_question)
        if qa_result['confidence'] > 0.3:
            context_parts.append(f"Q: {qa_result.get('matched_question', 'Похожий вопрос')}")
            context_parts.append(f"A: {qa_result['answer']}")
        
        # 2. Получаем связанные вопросы
        related_qa = self.qa_processor.get_related_questions(user_question, top_k=3)
        for qa in related_qa:
            context_parts.append(f"Q: {qa['question']}")
            context_parts.append(f"A: {qa['answer']}")
        
        # 3. Добавляем информацию о программах
        programs = self.db.get_all_programs()
        context_parts.append("\nИНФОРМАЦИЯ О ПРОГРАММАХ:")
        for program in programs:
            context_parts.append(f"\n📚 {program['name']}")
            context_parts.append(f"Описание: {program.get('description', '')[:200]}...")
            context_parts.append(f"Продолжительность: {program.get('duration', '')}")
            
            # Добавляем несколько курсов программы
            courses = self.db.get_courses_by_program(program['id'])[:5]
            if courses:
                context_parts.append("Примеры курсов:")
                for course in courses:
                    tags_str = ', '.join(course.get('tags', [])[:3])
                    context_parts.append(f"  - {course['name']} ({tags_str})")
        
        # 4. Если вопрос о конкретных курсах, добавляем релевантные курсы
        if any(word in user_question.lower() for word in ['курс', 'дисциплина', 'предмет', 'изучение']):
            all_courses = self.db.get_all_courses()
            
            # Поиск по ключевым словам
            course_keywords = ['машинное обучение', 'глубокое обучение', 'python', 'данные', 
                             'алгоритм', 'статистика', 'изображение', 'nlp', 'computer vision']
            
            relevant_courses = []
            for course in all_courses:
                course_text = f"{course['name']} {' '.join(course.get('tags', []))}".lower()
                for keyword in course_keywords:
                    if keyword in user_question.lower() and keyword in course_text:
                        relevant_courses.append(course)
                        break
            
            if relevant_courses:
                context_parts.append("\nСВЯЗАННЫЕ КУРСЫ:")
                for course in relevant_courses[:5]:
                    context_parts.append(f"  - {course['name']} ({course.get('program_name', 'Неизвестно')})")
                    if course.get('tags'):
                        context_parts.append(f"    Теги: {', '.join(course['tags'])}")
        
        return '\n'.join(context_parts)
    
    def generate_smart_answer(self, user_question: str, user_context: Dict = None) -> Dict:
        """Генерация умного ответа через GPT с использованием RAG"""
        
        if not self.is_available():
            # Fallback к базовой системе Q&A
            return self.qa_processor.get_answer(user_question)
        
        try:
            # Получаем контекст из базы знаний
            relevant_context = self.get_relevant_context(user_question)
            
            # Формируем сообщения для GPT
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
КОНТЕКСТ из базы знаний ИТМО:
{relevant_context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}

Ответь на вопрос, используя информацию из контекста. Если точной информации нет, скажи об этом и предложи альтернативы.
                """.strip()}
            ]
            
            # Добавляем информацию о пользователе если есть
            if user_context:
                user_info = f"Информация о пользователе: {json.dumps(user_context, ensure_ascii=False)}"
                messages[1]["content"] += f"\n\n{user_info}"
            
            # Запрос к GPT
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )
            
            gpt_answer = response.choices[0].message.content.strip()
            
            return {
                'answer': gpt_answer,
                'confidence': 0.9,  # Высокая уверенность для GPT ответов
                'method': 'gpt_rag',
                'context_used': len(relevant_context) > 100,
                'is_ai_generated': True
            }
            
        except Exception as e:
            logger.error(f"Ошибка GPT API: {e}")
            # Fallback к базовой системе
            fallback_result = self.qa_processor.get_answer(user_question)
            fallback_result['method'] = 'fallback_qa'
            return fallback_result
    
    def generate_course_recommendations_with_gpt(self, user_interests: str, 
                                               user_background: str = "") -> Dict:
        """Генерация рекомендаций курсов через GPT"""
        
        if not self.is_available():
            return {'error': 'GPT API недоступен'}
        
        try:
            # Получаем все курсы из базы
            all_courses = self.db.get_all_courses()
            
            # Формируем контекст с курсами
            courses_context = "ДОСТУПНЫЕ КУРСЫ ИТМО:\n\n"
            for course in all_courses:
                courses_context += f"📚 {course['name']}\n"
                courses_context += f"   Программа: {course.get('program_name', 'Неизвестно')}\n"
                courses_context += f"   Семестр: {course.get('semester', '')}\n"
                courses_context += f"   Кредиты: {course.get('credits', '')}\n"
                if course.get('tags'):
                    courses_context += f"   Теги: {', '.join(course['tags'])}\n"
                courses_context += "\n"
            
            messages = [
                {"role": "system", "content": """
Ты - эксперт по образовательным программам ИТМО в области ИИ. 
Рекомендуй курсы на основе интересов и бэкграунда пользователя.

ЗАДАЧА:
- Проанализируй интересы и опыт пользователя
- Выбери 5 наиболее подходящих курсов
- Объясни почему каждый курс подходит
- Дай советы по последовательности изучения

ФОРМАТ ОТВЕТА:
🎯 Персональные рекомендации:

🔥 1. [Название курса]
📚 Программа: [программа]
💡 Почему подходит: [обоснование]

[продолжи для всех 5 курсов]

💭 Советы по обучению: [общие рекомендации]
                """.strip()},
                {"role": "user", "content": f"""
{courses_context}

ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ:
Интересы: {user_interests}
Опыт/Бэкграунд: {user_background}

Рекомендуй наиболее подходящие курсы и объясни выбор.
                """.strip()}
            ]
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1500,
                temperature=0.8
            )
            
            recommendations = response.choices[0].message.content.strip()
            
            return {
                'recommendations': recommendations,
                'method': 'gpt_recommendations',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Ошибка при генерации рекомендаций через GPT: {e}")
            return {'error': str(e)}
    
    def get_program_comparison(self) -> Dict:
        """Сравнение программ через GPT"""
        
        if not self.is_available():
            return {'error': 'GPT API недоступен'}
        
        try:
            programs = self.db.get_all_programs()
            programs_context = ""
            
            for program in programs:
                programs_context += f"\n📚 {program['name']}\n"
                programs_context += f"Описание: {program.get('description', '')}\n"
                programs_context += f"Продолжительность: {program.get('duration', '')}\n"
                
                if program.get('career_prospects'):
                    programs_context += f"Карьерные перспективы: {', '.join(program['career_prospects'])}\n"
                
                # Добавляем примеры курсов
                courses = self.db.get_courses_by_program(program['id'])[:8]
                if courses:
                    programs_context += "Ключевые курсы:\n"
                    for course in courses:
                        programs_context += f"  - {course['name']}\n"
                programs_context += "\n" + "="*50 + "\n"
            
            messages = [
                {"role": "system", "content": """
Ты - консультант по образованию в ИТМО. Сравни программы магистратуры по ИИ.

ЗАДАЧА:
- Четко объясни различия между программами
- Для кого подходит каждая программа
- Ключевые преимущества каждой
- Помоги сделать выбор

ФОРМАТ:
🎓 Сравнение программ ИТМО:

📊 КЛЮЧЕВЫЕ РАЗЛИЧИЯ:
[основные отличия]

👨‍💼 "Искусственный интеллект" - для кого:
[целевая аудитория и особенности]

🚀 "AI-продукты" - для кого:
[целевая аудитория и особенности]

💡 Как выбрать:
[критерии выбора]
                """.strip()},
                {"role": "user", "content": f"""
Информация о программах ИТМО:
{programs_context}

Сделай подробное сравнение программ и дай рекомендации по выбору.
                """.strip()}
            ]
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            
            comparison = response.choices[0].message.content.strip()
            
            return {
                'comparison': comparison,
                'method': 'gpt_comparison',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении программ через GPT: {e}")
            return {'error': str(e)}

# Функции для альтернативных API
class FreeGPTIntegration(GPTIntegration):
    """Интеграция с бесплатными/альтернативными GPT API"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        super().__init__(db_manager)
        
        # Настройки для альтернативных API
        self.free_apis = [
            {
                'name': 'GPT4Free',
                'base_url': 'https://api.g4f.icu/v1',  # Пример бесплатного API
                'model': 'gpt-3.5-turbo'
            },
            {
                'name': 'OpenRouter',
                'base_url': 'https://openrouter.ai/api/v1',
                'model': 'openchat/openchat-7b:free'
            }
        ]
    
    def try_free_apis(self, messages: List[Dict]) -> Optional[str]:
        """Попытка использовать бесплатные API"""
        
        for api_config in self.free_apis:
            try:
                # Здесь можно добавить логику для конкретных API
                logger.info(f"Пробуем {api_config['name']}...")
                
                # Пример заглушки - в реальности здесь будет запрос к API
                # response = requests.post(api_config['base_url'] + '/chat/completions', ...)
                
                # Пока возвращаем None, чтобы использовать основной fallback
                return None
                
            except Exception as e:
                logger.warning(f"Ошибка {api_config['name']}: {e}")
                continue
        
        return None

if __name__ == "__main__":
    # Тестирование GPT интеграции
    gpt = GPTIntegration()
    
    if gpt.is_available():
        print("✅ GPT API доступен")
        
        test_question = "Какие курсы по машинному обучению есть в программе?"
        result = gpt.generate_smart_answer(test_question)
        
        print(f"\n❓ Вопрос: {test_question}")
        print(f"🤖 Ответ: {result['answer']}")
        print(f"📊 Метод: {result['method']}")
        
    else:
        print("❌ GPT API недоступен - работает только базовая система Q&A") 