import os
import logging
import requests
import json
from typing import Dict, List, Optional, Tuple

from src.database.db_manager import DatabaseManager
from src.nlp.qa_processor import QAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeGPTIntegration:
    """Интеграция с бесплатными GPT API"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()
        self.qa_processor = QAProcessor(self.db)
        
        # Настройки бесплатных API
        self.free_apis = [
            {
                'name': 'ProxyAPI',
                'base_url': 'https://api.proxyapi.ru/openai/v1',
                'model': 'gpt-3.5-turbo',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('PROXYAPI_KEY', '')}"
                },
                'auth_required': True,
                'key_env': 'PROXYAPI_KEY'
            },
            {
                'name': 'GPT4Free',
                'base_url': 'https://api.g4f.icu/v1',
                'model': 'gpt-3.5-turbo',
                'headers': {'Content-Type': 'application/json'},
                'auth_required': False
            },
            {
                'name': 'Groq',
                'base_url': 'https://api.groq.com/openai/v1',
                'model': 'llama3-8b-8192',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('GROQ_API_KEY', '')}"
                },
                'auth_required': True,
                'key_env': 'GROQ_API_KEY'
            },
            {
                'name': 'Together',
                'base_url': 'https://api.together.xyz/v1',
                'model': 'mistralai/Mistral-7B-Instruct-v0.1',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('TOGETHER_API_KEY', '')}"
                },
                'auth_required': True,
                'key_env': 'TOGETHER_API_KEY'
            },
            {
                'name': 'Hugging Face',
                'base_url': 'https://api-inference.huggingface.co/models',
                'model': 'microsoft/DialoGPT-large',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {os.getenv('HF_API_KEY', '')}"
                },
                'auth_required': True,
                'key_env': 'HF_API_KEY'
            }
        ]
        
        # Системный промпт для ИТМО бота  
        self.system_prompt = """
Ты - экспертный виртуальный консультант приемной комиссии университета ИТМО по магистерским программам в области искусственного интеллекта.

ТВОЯ РОЛЬ И ЭКСПЕРТИЗА:
• 🎓 Консультант по магистратуре ИТМО: "Искусственный интеллект" и "AI-продукты"
• 📚 Эксперт по учебным планам и выбору курсов  
• 💼 Консультант по карьерным траекториям и трудоустройству
• 🔍 Помощник в выборе специализации и построении индивидуального пути

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
✅ ИСПОЛЬЗУЙ ТОЛЬКО предоставленный контекст из базы знаний ИТМО
✅ Отвечай структурированно, подробно и профессионально  
✅ Для каждого ответа ссылайся на конкретные программы и курсы из контекста
✅ Используй эмодзи для лучшего восприятия информации
✅ Если точной информации нет в контексте - честно скажи об этом и направь к приемной комиссии

❌ НИКОГДА НЕ ВЫДУМЫВАЙ информацию о количестве мест, стоимости, датах поступления
❌ НЕ используй общие знания об образовании, только данные из предоставленного контекста  
❌ НЕ давай неточную информацию о поступлении без ссылки на контекст

СТРУКТУРА КАЧЕСТВЕННОГО ОТВЕТА:
1. 🎯 Прямой ответ на вопрос (если есть в контексте)
2. 📋 Детальная информация из базы знаний ИТМО  
3. 💡 Практические рекомендации и следующие шаги
4. 📞 Контакты для уточнения актуальной информации (если нужно)

КОНТЕКСТ из базы знаний ИТМО будет предоставлен ниже.
        """.strip()

    def is_available(self) -> bool:
        """Проверка доступности хотя бы одного API"""
        return len(self.get_available_apis()) > 0
    
    def get_available_apis(self) -> List[Dict]:
        """Получение списка доступных API"""
        available = []
        
        for api in self.free_apis:
            if not api.get('auth_required', False):
                available.append(api)
            elif api.get('key_env') and os.getenv(api['key_env']):
                available.append(api)
        
        return available

    def call_proxyapi(self, messages: List[Dict]) -> Optional[str]:
        """Вызов ProxyAPI"""
        try:
            api_key = os.getenv('PROXYAPI_KEY')
            if not api_key:
                logger.warning("🚫 ProxyAPI: ключ не найден в окружении")
                return None
            
            logger.info(f"🔑 ProxyAPI: используем ключ {api_key[:20]}...")
            
            url = "https://api.proxyapi.ru/openai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False
            }
            
            logger.info(f"📤 ProxyAPI: отправляем запрос к {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            logger.info(f"📥 ProxyAPI: получен ответ со статусом {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    answer = data['choices'][0]['message']['content']
                    logger.info(f"✅ ProxyAPI: получен ответ длиной {len(answer)} символов")
                    return answer
                else:
                    logger.warning(f"❌ ProxyAPI: неожиданный формат ответа: {data}")
            else:
                logger.warning(f"❌ ProxyAPI ошибка HTTP {response.status_code}: {response.text[:200]}")
            
            return None
            
        except Exception as e:
            logger.error(f"💥 Исключение в ProxyAPI: {type(e).__name__}: {e}")
            return None

    def call_gpt4free(self, messages: List[Dict]) -> Optional[str]:
        """Вызов GPT4Free API"""
        try:
            url = "https://api.g4f.icu/v1/chat/completions"
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
            
            logger.warning(f"GPT4Free ошибка: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка GPT4Free: {e}")
            return None

    def call_groq(self, messages: List[Dict]) -> Optional[str]:
        """Вызов Groq API"""
        try:
            if not os.getenv('GROQ_API_KEY'):
                return None
            
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
            
            logger.warning(f"Groq ошибка: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка Groq: {e}")
            return None

    def call_together(self, messages: List[Dict]) -> Optional[str]:
        """Вызов Together AI API"""
        try:
            if not os.getenv('TOGETHER_API_KEY'):
                return None
            
            url = "https://api.together.xyz/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
            
            logger.warning(f"Together ошибка: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка Together: {e}")
            return None

    def try_free_apis(self, messages: List[Dict]) -> Optional[str]:
        """Попытка использовать бесплатные API по порядку"""
        
        # Сначала пробуем ProxyAPI (если есть ключ)
        proxyapi_key = os.getenv('PROXYAPI_KEY')
        logger.info(f"🔍 Проверяем ProxyAPI ключ: {'есть' if proxyapi_key else 'отсутствует'}")
        
        if proxyapi_key:
            logger.info("🔄 Пробуем ProxyAPI...")
            result = self.call_proxyapi(messages)
            if result:
                logger.info("✅ ProxyAPI успешно ответил")
                return result
            else:
                logger.warning("❌ ProxyAPI не ответил")
        else:
            logger.info("⏭️ Пропускаем ProxyAPI - ключ не найден")
        
        # Потом пробуем GPT4Free (не требует ключа)
        logger.info("🔄 Пробуем GPT4Free...")
        result = self.call_gpt4free(messages)
        if result:
            logger.info("✅ GPT4Free успешно ответил")
            return result
        else:
            logger.warning("❌ GPT4Free не ответил")
        
        # Затем пробуем Groq (быстрый и качественный)
        if os.getenv('GROQ_API_KEY'):
            logger.info("Пробуем Groq...")
            result = self.call_groq(messages)
            if result:
                logger.info("✅ Groq успешно ответил")
                return result
        
        # Затем Together AI с Mistral
        if os.getenv('TOGETHER_API_KEY'):
            logger.info("Пробуем Together AI...")
            result = self.call_together(messages)
            if result:
                logger.info("✅ Together AI успешно ответил")
                return result
        
        logger.warning("❌ Все бесплатные API недоступны")
        return None

    def get_relevant_context(self, user_question: str, max_items: int = 10) -> str:
        """Получение релевантного контекста из базы знаний"""
        context_parts = []
        question_lower = user_question.lower()
        
        # 1. Ищем похожие Q&A
        qa_result = self.qa_processor.get_answer(user_question)
        if qa_result['confidence'] > 0.2:
            context_parts.append(f"Q: {qa_result.get('matched_question', 'Похожий вопрос')}")
            context_parts.append(f"A: {qa_result['answer']}")
        
        # 2. Получаем связанные вопросы
        related_qa = self.qa_processor.get_related_questions(user_question, top_k=5)
        if related_qa:
            context_parts.append("\n=== ПОХОЖИЕ ВОПРОСЫ ===")
            for qa in related_qa:
                context_parts.append(f"Q: {qa['question']}")
                context_parts.append(f"A: {qa['answer']}")
        
        # 3. ДЕТАЛЬНАЯ информация о программах
        programs = self.db.get_all_programs()
        context_parts.append("\n=== ПОЛНАЯ ИНФОРМАЦИЯ О ПРОГРАММАХ ИТМО ===")
        
        for program in programs:
            context_parts.append(f"\n🎓 ПРОГРАММА: {program['name']}")
            context_parts.append(f"📝 Описание: {program.get('description', '')}")
            context_parts.append(f"⏱️ Продолжительность: {program.get('duration', '2 года')}")
            context_parts.append(f"🎯 Уровень: {program.get('level', 'Магистратура')}")
            
            # Добавляем БОЛЬШЕ курсов программы
            courses = self.db.get_courses_by_program(program['id'])
            if courses:
                context_parts.append(f"📚 Курсы программы ({len(courses)} курсов):")
                for course in courses[:15]:  # Увеличиваем до 15 курсов
                    tags_str = ', '.join(course.get('tags', [])[:5])
                    is_mandatory = "ОБЯЗАТЕЛЬНЫЙ" if course.get('is_mandatory') else "ВЫБОРНЫЙ"
                    context_parts.append(f"  • {course['name']} [{is_mandatory}] (Теги: {tags_str})")
            
            # Добавляем карьерную информацию если есть
            if program.get('career_info'):
                context_parts.append(f"💼 Карьера: {program['career_info']}")
            
            context_parts.append("---")
        
        # 4. СПЕЦИАЛЬНАЯ информация для вопросов о поступлении
        admission_keywords = ['бюджет', 'мест', 'поступление', 'требования', 'экзамен', 'стоимость', 'цена']
        if any(keyword in question_lower for keyword in admission_keywords):
            context_parts.append("\n=== ИНФОРМАЦИЯ О ПОСТУПЛЕНИИ ===")
            context_parts.append("📊 Статистика поступления в ИТМО:")
            context_parts.append("• Магистратура по ИИ - высокий конкурс")
            context_parts.append("• Требования: портфолио, собеседование, мотивационное письмо")
            context_parts.append("• Форма обучения: очная, 2 года")
            context_parts.append("• Язык обучения: русский/английский")
            context_parts.append("⚠️ ВНИМАНИЕ: Точное количество бюджетных мест и стоимость обучения")
            context_parts.append("   уточняйте в приемной комиссии ИТМО, так как эта информация")
            context_parts.append("   изменяется каждый год и зависит от государственного заказа.")
        
        # 5. Если вопрос о конкретных курсах, добавляем релевантные курсы  
        if any(word in question_lower for word in ['курс', 'дисциплина', 'предмет', 'изучение']):
            all_courses = self.db.get_all_courses()
            
            # Поиск по ключевым словам
            course_keywords = ['машинное обучение', 'глубокое обучение', 'python', 'данные', 
                             'алгоритм', 'статистика', 'изображение', 'nlp', 'computer vision',
                             'рекомендательные системы', 'веб-разработка', 'программирование']
            
            relevant_courses = []
            for course in all_courses:
                course_text = f"{course['name']} {' '.join(course.get('tags', []))}".lower()
                for keyword in course_keywords:
                    if keyword in question_lower and keyword in course_text:
                        relevant_courses.append(course)
                        break
            
            if relevant_courses:
                context_parts.append("\n=== СВЯЗАННЫЕ КУРСЫ ===")
                for course in relevant_courses[:10]:
                    context_parts.append(f"📖 {course['name']} ({course.get('program_name', 'Неизвестно')})")
                    if course.get('tags'):
                        context_parts.append(f"   Теги: {', '.join(course['tags'])}")
        
        # 6. Добавляем общую статистику
        total_programs = len(programs)
        total_courses = len(self.db.get_all_courses()) if hasattr(self.db, 'get_all_courses') else 0
        context_parts.append(f"\n=== ОБЩАЯ СТАТИСТИКА ===")
        context_parts.append(f"📊 Всего программ: {total_programs}")
        context_parts.append(f"📚 Всего курсов: {total_courses}")
        context_parts.append(f"🏫 Университет: ИТМО (Санкт-Петербург)")
        
        return '\n'.join(context_parts)

    def generate_smart_answer(self, user_question: str, user_context: Dict = None) -> Dict:
        """Генерация умного ответа через бесплатные GPT API с использованием RAG"""
        
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
            
            # Пробуем бесплатные API
            gpt_answer = self.try_free_apis(messages)
            
            if gpt_answer:
                return {
                    'answer': gpt_answer,
                    'confidence': 0.9,  # Высокая уверенность для GPT ответов
                    'method': 'free_gpt_rag',
                    'context_used': len(relevant_context) > 100,
                    'is_ai_generated': True
                }
            else:
                # Fallback к базовой системе
                fallback_result = self.qa_processor.get_answer(user_question)
                fallback_result['method'] = 'fallback_qa'
                return fallback_result
            
        except Exception as e:
            logger.error(f"Ошибка Free GPT API: {e}")
            # Fallback к базовой системе
            fallback_result = self.qa_processor.get_answer(user_question)
            fallback_result['method'] = 'fallback_qa'
            return fallback_result

    def get_program_comparison(self) -> Dict:
        """Сравнение программ через бесплатные GPT API"""
        
        if not self.is_available():
            return {'error': 'Бесплатные GPT API недоступны'}
        
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

Отвечай на русском языке, используй эмодзи.
                """.strip()},
                {"role": "user", "content": f"""
Информация о программах ИТМО:
{programs_context}

Сделай подробное сравнение программ и дай рекомендации по выбору.
                """.strip()}
            ]
            
            comparison = self.try_free_apis(messages)
            
            if comparison:
                return {
                    'comparison': comparison,
                    'method': 'free_gpt_comparison',
                    'success': True
                }
            else:
                return {'error': 'Не удалось получить ответ от бесплатных API'}
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении программ через Free GPT: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Тестирование Free GPT интеграции
    free_gpt = FreeGPTIntegration()
    
    print("🆓 Тестирование бесплатных GPT API")
    print("=" * 50)
    
    available_apis = free_gpt.get_available_apis()
    print(f"📊 Доступно API: {len(available_apis)}")
    for api in available_apis:
        print(f"  ✅ {api['name']} - {api['model']}")
    
    if free_gpt.is_available():
        print("\n🤖 Тестируем ответ...")
        
        test_question = "Какие курсы по машинному обучению есть в программе?"
        result = free_gpt.generate_smart_answer(test_question)
        
        print(f"\n❓ Вопрос: {test_question}")
        print(f"🤖 Ответ: {result['answer'][:200]}...")
        print(f"📊 Метод: {result['method']}")
        
    else:
        print("❌ Бесплатные GPT API недоступны") 