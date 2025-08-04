import logging
from typing import Dict, List, Optional, Tuple
import re

from src.database.db_manager import DatabaseManager
from src.nlp.qa_processor import QAProcessor
from src.nlp.course_recommender import CourseRecommender
from src.nlp.free_gpt_integration import FreeGPTIntegration
from src.nlp.smart_qa_processor import SmartQAProcessor
from config import ENABLE_GPT_MODE, GPT_MODE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotHandler:
    def __init__(self):
        """Инициализация обработчика бота"""
        self.db = DatabaseManager()
        self.qa_processor = QAProcessor(self.db)
        self.recommender = CourseRecommender(self.db)
        
        # Бесплатная GPT интеграция
        self.gpt = FreeGPTIntegration(self.db) if ENABLE_GPT_MODE else None
        self.gpt_available = self.gpt and self.gpt.is_available()
        
        # Локальная умная система (всегда доступна)
        self.smart_qa = SmartQAProcessor(self.db)
        
        if self.gpt_available:
            available_apis = self.gpt.get_available_apis()
            api_names = [api['name'] for api in available_apis]
            logger.info(f"🆓 Бесплатные GPT API включены: {', '.join(api_names)}")
        
        logger.info("🧠 Локальная умная система активна")
        
        # Шаблонные команды и их обработчики
        self.command_handlers = {
            '/start': self.handle_start,
            '/help': self.handle_help,
            '/programs': self.handle_programs,
            '/courses': self.handle_courses,
            '/recommend': self.handle_recommend,
            '/profile': self.handle_profile,
            '/faq': self.handle_faq,
            '/gpt': self.handle_gpt_mode,  # Умный режим (внешний GPT или локальный)
            '/smart': self.handle_smart_mode,  # Локальная умная система
            '/compare': self.handle_program_comparison  # Сравнение программ
        }
        
        # Состояния пользователя для многошагового диалога
        self.user_states = {}
        
        # Шаблонные вопросы и быстрые ответы
        self.quick_questions = {
            '📚 Какие программы доступны?': self.handle_programs,
            '⏱️ Сколько длится обучение?': lambda user_id, username: self.handle_duration_question(),
            '🎯 Требования для поступления?': lambda user_id, username: self.handle_admission_question(),
            '💼 Карьерные перспективы?': lambda user_id, username: self.handle_career_question(),
            '🔍 Получить рекомендации': self.handle_recommend,
            '❓ Задать вопрос': lambda user_id, username: self.handle_ask_question_mode(user_id),
            # Добавляем недостающие кнопки
            '🔄 Обновить профиль': self.handle_profile,
            '📚 Все курсы': self.handle_courses,
            '🎓 О программах': self.handle_programs,
            '🔙 Главное меню': self.handle_start,
            '📖 Посмотреть курсы': self.handle_courses,
            '⚙️ Настроить профиль': self.handle_profile,
            # Умные режимы
            '🤖 Умный ответ': self.handle_gpt_mode,
            '🧠 Умный ответ': self.handle_smart_mode,
            '📊 Сравнить программы': self.handle_program_comparison
        }
    
    def process_message(self, user_id: int, username: str, message: str) -> Dict:
        """Основная функция обработки сообщений"""
        try:
            # Нормализуем сообщение
            message = message.strip()
            
            # Проверяем, есть ли пользователь в базе
            user = self.db.get_user_by_telegram_id(user_id)
            if not user:
                self.db.insert_user(user_id, username)
                user = self.db.get_user_by_telegram_id(user_id)
            
            # Обрабатываем команды
            if message.startswith('/'):
                return self.handle_command(user_id, username, message)
            
            # Обрабатываем быстрые вопросы
            if message in self.quick_questions:
                handler = self.quick_questions[message]
                return handler(user_id, username)
            
            # Проверяем состояние пользователя для многошагового диалога
            user_state = self.user_states.get(user_id, {})
            if user_state.get('state'):
                return self.handle_state_message(user_id, username, message, user_state)
            
            # Обрабатываем обычный вопрос через Q&A процессор
            return self.handle_question(user_id, username, message)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения от {user_id}: {e}")
            return {
                'text': 'Извините, произошла ошибка. Попробуйте позже или обратитесь к администратору.',
                'keyboard': self.get_main_keyboard()
            }
    
    def handle_command(self, user_id: int, username: str, command: str) -> Dict:
        """Обработка команд"""
        handler = self.command_handlers.get(command)
        if handler:
            return handler(user_id, username)
        else:
            return {
                'text': f'Неизвестная команда: {command}\nИспользуйте /help для просмотра доступных команд.',
                'keyboard': self.get_main_keyboard()
            }
    
    def handle_start(self, user_id: int, username: str) -> Dict:
        """Обработка команды /start"""
        welcome_text = f"""
👋 Привет{f', {username}' if username else ''}!

Я бот-помощник для абитуриентов магистерских программ ИТМО по искусственному интеллекту.

🎯 Что я умею:
• Отвечать на вопросы о программах обучения
• Рекомендовать курсы на основе ваших интересов
• Помогать выбрать между программами "AI" и "AI-продукты"
• Предоставлять информацию о поступлении и карьерных перспективах

📝 Доступные программы:
• Искусственный интеллект (фундаментальные исследования)
• AI-продукты (коммерческая разработка)

Выберите интересующую тему или задайте свой вопрос!
        """
        
        return {
            'text': welcome_text.strip(),
            'keyboard': self.get_main_keyboard()
        }
    
    def handle_help(self, user_id: int, username: str) -> Dict:
        """Обработка команды /help"""
        help_text = """
🤖 Доступные команды:

/start - Начать работу с ботом
/help - Показать это сообщение
/programs - Информация о программах
/courses - Список курсов
/recommend - Получить рекомендации курсов
/profile - Настроить профиль для рекомендаций
/faq - Часто задаваемые вопросы

📝 Как пользоваться:
1. Выберите тему из кнопок меню
2. Или напишите свой вопрос в свободной форме
3. Для получения персональных рекомендаций сначала настройте профиль

💡 Примеры вопросов:
• "Какие курсы по машинному обучению есть?"
• "В чем разница между программами?"
• "Сколько длится обучение?"
• "Какие требования для поступления?"
        """
        
        return {
            'text': help_text.strip(),
            'keyboard': self.get_main_keyboard()
        }
    
    def handle_programs(self, user_id: int, username: str) -> Dict:
        """Обработка информации о программах"""
        programs = self.db.get_all_programs()
        
        if not programs:
            return {
                'text': 'К сожалению, информация о программах временно недоступна.',
                'keyboard': self.get_main_keyboard()
            }
        
        response_text = "🎓 Доступные магистерские программы:\n\n"
        
        for program in programs:
            courses_count = len(self.db.get_courses_by_program(program['id']))
            
            response_text += f"📋 {program['name']}\n"
            response_text += f"⏱️ Продолжительность: {program.get('duration', '2 года')}\n"
            response_text += f"📚 Количество курсов: {courses_count}\n"
            response_text += f"📝 {program.get('description', 'Описание отсутствует')[:200]}...\n"
            
            if program.get('career_prospects'):
                prospects = program['career_prospects'][:3]  # Первые 3 перспективы
                response_text += f"💼 Карьера: {', '.join(prospects)}\n"
            
            response_text += "\n" + "─" * 30 + "\n\n"
        
        keyboard = self.get_programs_keyboard()
        
        return {
            'text': response_text.strip(),
            'keyboard': keyboard
        }
    
    def handle_courses(self, user_id: int, username: str) -> Dict:
        """Обработка запроса курсов"""
        # Предлагаем выбрать программу для просмотра курсов
        programs = self.db.get_all_programs()
        
        if not programs:
            return {
                'text': 'К сожалению, информация о курсах временно недоступна.',
                'keyboard': self.get_main_keyboard()
            }
        
        # Устанавливаем состояние для выбора программы
        self.user_states[user_id] = {'state': 'select_program_for_courses'}
        
        keyboard = []
        for program in programs:
            courses_count = len(self.db.get_courses_by_program(program['id']))
            keyboard.append([f"📚 {program['name']} ({courses_count} курсов)"])
        
        keyboard.append(["🔙 Назад в главное меню"])
        
        return {
            'text': '📚 Выберите программу для просмотра курсов:',
            'keyboard': keyboard
        }
    
    def handle_recommend(self, user_id: int, username: str) -> Dict:
        """Обработка запроса рекомендаций"""
        # Проверяем, есть ли информация о пользователе
        user = self.db.get_user_by_telegram_id(user_id)
        
        if not user or not user.get('interests'):
            # Предлагаем настроить профиль
            self.user_states[user_id] = {'state': 'setup_profile_for_recommendations'}
            
            return {
                'text': """
🎯 Персональные рекомендации курсов

Для получения персональных рекомендаций расскажите о своих интересах и опыте.

📝 Напишите:
• Ваш опыт в программировании (языки, фреймворки)
• Интересующие области (ML, CV, NLP, Data Science и т.д.)
• Цели обучения (исследования, коммерческая разработка)

💡 Пример:
"Знаю Python, работал с pandas и sklearn. Интересует машинное обучение и computer vision. Хочу заниматься исследованиями."
                """.strip(),
                'keyboard': [["🔙 Отмена"]]
            }
        
        # Если профиль уже настроен, даем рекомендации
        return self.generate_recommendations(user_id, user)
    
    def handle_profile(self, user_id: int, username: str) -> Dict:
        """Обработка настройки профиля"""
        user = self.db.get_user_by_telegram_id(user_id)
        
        current_info = ""
        if user and user.get('interests'):
            current_info = f"\n🔍 Текущие интересы: {', '.join(user['interests'])}"
        if user and user.get('background'):
            bg = user['background']
            current_info += f"\n💼 Опыт: {bg.get('experience_level', 'Не указан')}"
            if bg.get('technical_skills'):
                current_info += f"\n⚙️ Навыки: {', '.join(bg['technical_skills'])}"
        
        self.user_states[user_id] = {'state': 'update_profile'}
        
        text = f"""
👤 Настройка профиля{current_info}

📝 Расскажите о себе, чтобы получать персональные рекомендации курсов:

• Ваш опыт в программировании
• Интересующие области ИИ
• Цели обучения
• Предпочитаемая программа (если есть)

💡 Пример:
"Программирую на Python 3 года, работаю Data Scientist. Интересует deep learning и NLP. Хочу углубить знания для карьеры в AI research. Склоняюсь к программе Искусственный интеллект."
        """
        
        return {
            'text': text.strip(),
            'keyboard': [["🔙 Отмена"]]
        }
    
    def handle_faq(self, user_id: int, username: str) -> Dict:
        """Обработка FAQ"""
        qa_pairs = self.db.get_all_qa_pairs()
        
        if not qa_pairs:
            return {
                'text': 'База вопросов и ответов пуста.',
                'keyboard': self.get_main_keyboard()
            }
        
        # Группируем по категориям
        categories = {}
        for qa in qa_pairs:
            category = qa.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append(qa)
        
        response_text = "❓ Часто задаваемые вопросы:\n\n"
        
        category_names = {
            'general': '📋 Общие вопросы',
            'admission': '🎓 Поступление',
            'programs': '📚 О программах',
            'courses': '📖 Курсы',
            'career': '💼 Карьера'
        }
        
        for category, questions in categories.items():
            category_name = category_names.get(category, category.title())
            response_text += f"{category_name}\n"
            
            for qa in questions[:3]:  # Показываем по 3 вопроса в категории
                response_text += f"❔ {qa['question']}\n"
                response_text += f"💬 {qa['answer'][:100]}...\n\n"
            
            response_text += "─" * 30 + "\n\n"
        
        response_text += "💡 Можете задать любой вопрос в свободной форме!"
        
        return {
            'text': response_text.strip(),
            'keyboard': self.get_main_keyboard()
        }
    
    def handle_question(self, user_id: int, username: str, question: str) -> Dict:
        """Обработка свободного вопроса"""
        # Сначала пробуем базовую Q&A систему
        result = self.qa_processor.get_answer(question)
        
        logger.info(f"🔍 Базовая Q&A система: confidence={result['confidence']:.3f}, is_exact_match={result.get('is_exact_match', False)}")
        logger.info(f"🔧 GPT настройки: gpt_available={self.gpt_available}, threshold={GPT_MODE_THRESHOLD}")
        
        # Если GPT доступен и базовый ответ имеет низкое качество, используем внешний GPT
        if (self.gpt_available and 
            result['confidence'] < GPT_MODE_THRESHOLD and 
            not result.get('is_exact_match', False)):
            
            logger.info(f"✅ Переключение на внешний GPT режим для вопроса: {question}")
            
            # Получаем контекст пользователя
            user = self.db.get_user_by_telegram_id(user_id)
            user_context = None
            if user:
                user_context = {
                    'interests': user.get('interests', []),
                    'background': user.get('background', {}),
                    'preferred_program': user.get('preferred_program')
                }
            
            # Генерируем умный ответ через внешний GPT
            gpt_result = self.gpt.generate_smart_answer(question, user_context)
            
            if gpt_result.get('is_ai_generated'):
                response_text = gpt_result['answer']
                response_text += f"\n\n🆓 Ответ от бесплатного GPT с базой знаний ИТМО"
                
                return {
                    'text': response_text,
                    'keyboard': self.get_gpt_keyboard()
                }
        else:
            logger.info(f"❌ Внешний GPT пропущен: gpt_available={self.gpt_available}, confidence={result['confidence']:.3f}, threshold={GPT_MODE_THRESHOLD}, is_exact_match={result.get('is_exact_match', False)}")
        
        # Если внешний GPT недоступен или дал плохой результат, используем локальную умную систему
        if result['confidence'] < GPT_MODE_THRESHOLD:
            logger.info(f"Переключение на локальную умную систему для вопроса: {question}")
            
            # Получаем контекст пользователя
            user = self.db.get_user_by_telegram_id(user_id)
            user_context = None
            if user:
                user_context = {
                    'interests': user.get('interests', []),
                    'background': user.get('background', {}),
                    'preferred_program': user.get('preferred_program')
                }
            
            smart_result = self.smart_qa.generate_smart_answer(question, user_context)
            
            if smart_result.get('is_ai_generated') or smart_result.get('method') in ['smart_enhanced', 'smart_local']:
                response_text = smart_result['answer']
                response_text += f"\n\n🧠 Умный анализ на основе базы знаний ИТМО"
                
                return {
                    'text': response_text,
                    'keyboard': self.get_smart_keyboard()
                }
        
        # Используем обычный ответ из базы знаний
        response_text = result['answer']
        
        # Добавляем информацию о качестве ответа
        if result['confidence'] < 0.5:
            response_text += "\n\n💡 Если ответ не подходит, попробуйте переформулировать вопрос."
            response_text += "\n🧠 Или используйте кнопку 'Умный ответ' для детального анализа."
        
        # Предлагаем связанные вопросы если есть
        if result['confidence'] > 0.3:
            related = self.qa_processor.get_related_questions(question, top_k=2)
            if related:
                response_text += "\n\n📋 Возможно, вас также интересует:\n"
                for rel in related:
                    response_text += f"• {rel['question']}\n"
        
        keyboard = self.get_main_keyboard()
        if result['confidence'] < 0.7:
            keyboard = self.get_smart_keyboard()
        
        return {
            'text': response_text,
            'keyboard': keyboard
        }
    
    def handle_state_message(self, user_id: int, username: str, message: str, user_state: Dict) -> Dict:
        """Обработка сообщений в рамках многошагового диалога"""
        state = user_state['state']
        
        if message == "🔙 Отмена" or message == "🔙 Назад в главное меню":
            # Отменяем текущее состояние
            self.user_states.pop(user_id, None)
            return self.handle_start(user_id, username)
        
        if state == 'setup_profile_for_recommendations' or state == 'update_profile':
            return self.handle_profile_input(user_id, message)
        
        elif state == 'select_program_for_courses':
            return self.handle_program_selection_for_courses(user_id, message)
        
        elif state == 'gpt_mode':
            # GPT режим - отвечаем через внешний или локальный GPT
            return self.handle_gpt_question(user_id, username, message)
        
        elif state == 'smart_mode':
            # Локальная умная система
            return self.handle_smart_question(user_id, username, message)
        
        elif state == 'program_comparison':
            return self.handle_program_comparison_gpt(user_id, message)
        
        # Сбрасываем состояние если не обработали
        self.user_states.pop(user_id, None)
        return self.handle_question(user_id, username, message)
    
    def handle_profile_input(self, user_id: int, profile_text: str) -> Dict:
        """Обработка ввода профиля пользователя"""
        try:
            # Анализируем текст пользователя
            analysis = self.recommender.analyze_user_background(profile_text)
            interests = self.recommender.extract_interests_from_text(profile_text)
            
            # Определяем предпочитаемую программу
            preferred_program = None
            text_lower = profile_text.lower()
            if 'ai-продукт' in text_lower or 'продукт' in text_lower:
                preferred_program = 'AI_Product'
            elif 'искусственный интеллект' in text_lower or 'исследован' in text_lower:
                preferred_program = 'AI'
            
            # Сохраняем в базу данных
            self.db.update_user_preferences(
                user_id,
                background=analysis,
                interests=list(interests.keys()) if interests else [],
                preferred_program=preferred_program
            )
            
            # Сбрасываем состояние
            self.user_states.pop(user_id, None)
            
            # Генерируем рекомендации
            user = self.db.get_user_by_telegram_id(user_id)
            recommendations_response = self.generate_recommendations(user_id, user)
            
            # Добавляем информацию о профиле
            profile_info = f"""
✅ Профиль обновлен!

🔍 Обнаруженные интересы: {', '.join(interests.keys()) if interests else 'Не определены'}
💼 Уровень опыта: {analysis.get('experience_level', 'Не определен')}
⚙️ Технические навыки: {', '.join(analysis.get('technical_skills', [])) or 'Не определены'}
            """
            
            if preferred_program:
                program_name = 'AI-продукты' if preferred_program == 'AI_Product' else 'Искусственный интеллект'
                profile_info += f"\n🎯 Предпочитаемая программа: {program_name}"
            
            profile_info += "\n\n" + "─" * 30 + "\n\n"
            
            recommendations_response['text'] = profile_info + recommendations_response['text']
            
            return recommendations_response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке профиля: {e}")
            self.user_states.pop(user_id, None)
            return {
                'text': 'Произошла ошибка при обработке профиля. Попробуйте еще раз.',
                'keyboard': self.get_main_keyboard()
            }
    
    def generate_recommendations(self, user_id: int, user: Dict) -> Dict:
        """Генерация рекомендаций для пользователя"""
        try:
            interests = {}
            if user.get('interests'):
                # Преобразуем список интересов в словарь с весами
                for interest in user['interests']:
                    interests[interest] = 1.0
            
            preferred_program = user.get('preferred_program')
            
            # Получаем рекомендации
            recommendations = self.recommender.recommend_courses(
                interests, 
                preferred_program, 
                top_k=5
            )
            
            if not recommendations:
                return {
                    'text': 'К сожалению, не удалось найти подходящие курсы. Попробуйте обновить профиль или задать вопрос администратору.',
                    'keyboard': self.get_main_keyboard()
                }
            
            # Сохраняем рекомендации
            self.recommender.save_recommendations(user_id, recommendations)
            
            # Формируем ответ
            response_text = "🎯 Персональные рекомендации курсов:\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                score_emoji = "🔥" if rec['score'] > 0.7 else "👍" if rec['score'] > 0.4 else "💡"
                mandatory_mark = " ⭐ (обязательный)" if rec['is_mandatory'] else ""
                
                response_text += f"{score_emoji} {i}. {rec['course_name']}{mandatory_mark}\n"
                response_text += f"📚 Программа: {rec['program_name']}\n"
                response_text += f"📅 Семестр: {rec['semester']} | 💳 Кредиты: {rec['credits']}\n"
                response_text += f"🎯 Релевантность: {rec['score']:.0%}\n"
                response_text += f"💡 Обоснование: {rec['reason']}\n\n"
            
            response_text += "💡 Рекомендации основаны на вашем профиле. Обновите профиль для более точных рекомендаций!"
            
            keyboard = [
                ["🔄 Обновить профиль", "📚 Все курсы"],
                ["🎓 О программах", "🔙 Главное меню"]
            ]
            
            return {
                'text': response_text.strip(),
                'keyboard': keyboard
            }
            
        except Exception as e:
            logger.error(f"Ошибка при генерации рекомендаций: {e}")
            return {
                'text': 'Произошла ошибка при генерации рекомендаций. Попробуйте позже.',
                'keyboard': self.get_main_keyboard()
            }
    
    def handle_program_selection_for_courses(self, user_id: int, message: str) -> Dict:
        """Обработка выбора программы для просмотра курсов"""
        # Извлекаем название программы из сообщения
        if "Искусственный интеллект" in message:
            program_name = "Искусственный интеллект"
        elif "AI-продукты" in message:
            program_name = "AI-продукты"
        else:
            self.user_states.pop(user_id, None)
            return {
                'text': 'Неверный выбор программы.',
                'keyboard': self.get_main_keyboard()
            }
        
        # Находим программу в базе
        programs = self.db.get_all_programs()
        program = None
        for p in programs:
            if p['name'] == program_name:
                program = p
                break
        
        if not program:
            self.user_states.pop(user_id, None)
            return {
                'text': 'Программа не найдена.',
                'keyboard': self.get_main_keyboard()
            }
        
        # Получаем курсы
        courses = self.db.get_courses_by_program(program['id'])
        
        # Сбрасываем состояние
        self.user_states.pop(user_id, None)
        
        if not courses:
            return {
                'text': f'Курсы для программы "{program_name}" не найдены.',
                'keyboard': self.get_main_keyboard()
            }
        
        # Группируем курсы по семестрам
        semesters = {}
        for course in courses:
            semester = course.get('semester', 'Неизвестный семестр')
            if semester not in semesters:
                semesters[semester] = []
            semesters[semester].append(course)
        
        response_text = f"📚 Курсы программы '{program_name}':\n\n"
        
        for semester in sorted(semesters.keys()):
            semester_courses = semesters[semester]
            response_text += f"📅 {semester}\n"
            
            for course in semester_courses[:10]:  # Показываем до 10 курсов за семестр
                mandatory_mark = " ⭐" if course.get('is_mandatory') else ""
                tags_text = f" | {', '.join(course.get('tags', [])[:3])}" if course.get('tags') else ""
                
                response_text += f"• {course['name']}{mandatory_mark} ({course.get('credits', 0)} кр.){tags_text}\n"
            
            if len(semester_courses) > 10:
                response_text += f"... и еще {len(semester_courses) - 10} курсов\n"
            
            response_text += "\n"
        
        response_text += f"📊 Всего курсов: {len(courses)}\n"
        response_text += "⭐ - обязательные курсы\n\n"
        response_text += "💡 Для персональных рекомендаций используйте команду /recommend"
        
        return {
            'text': response_text.strip(),
            'keyboard': self.get_main_keyboard()
        }
    
    # Вспомогательные методы для стандартных ответов
    def handle_duration_question(self) -> Dict:
        """Ответ на вопрос о продолжительности"""
        return {
            'text': '⏱️ Продолжительность обучения: 2 года (4 семестра)\n\nЭто стандартная продолжительность магистерских программ в ИТМО.',
            'keyboard': self.get_main_keyboard()
        }
    
    def handle_admission_question(self) -> Dict:
        """Ответ на вопрос о требованиях к поступлению"""
        programs = self.db.get_all_programs()
        if programs and programs[0].get('admission_requirements'):
            requirements = programs[0]['admission_requirements']
            req_text = '\n• '.join([''] + requirements)
        else:
            req_text = """
• Высшее образование (диплом бакалавра или специалиста)
• Вступительные испытания по профилю программы
• Конкурсный отбор по результатам вступительных испытаний"""
        
        return {
            'text': f'🎓 Требования для поступления:{req_text}',
            'keyboard': self.get_main_keyboard()
        }
    
    def handle_career_question(self) -> Dict:
        """Ответ на вопрос о карьерных перспективах"""
        programs = self.db.get_all_programs()
        all_prospects = set()
        
        for program in programs:
            if program.get('career_prospects'):
                all_prospects.update(program['career_prospects'])
        
        if all_prospects:
            prospects_text = '\n• '.join([''] + list(all_prospects))
        else:
            prospects_text = """
• ML-инженер
• Data Scientist  
• AI-разработчик
• Исследователь в области ИИ
• Продуктовый менеджер AI-продуктов"""
        
        return {
            'text': f'💼 Карьерные перспективы:{prospects_text}',
            'keyboard': self.get_main_keyboard()
        }
    
    def handle_ask_question_mode(self, user_id: int) -> Dict:
        """Переход в режим вопросов"""
        return {
            'text': '❓ Задайте свой вопрос\n\nНапишите любой вопрос о программах, курсах, поступлении или карьерных перспективах.',
            'keyboard': [["🔙 Назад в главное меню"]]
        }
    
    def handle_gpt_mode(self, user_id: int, username: str) -> Dict:
        """Обработка команды /gpt"""
        if not self.gpt_available:
            return {
                'text': 'GPT режим временно недоступен. Пожалуйста, попробуйте позже.',
                'keyboard': self.get_main_keyboard()
            }
        
        self.user_states[user_id] = {'state': 'gpt_mode'}
        return {
            'text': '🤖 Вы перешли в умный режим!\n\nТеперь я буду отвечать на ваши вопросы, используя продвинутый ИИ и базу знаний ИТМО.\n\nЗадайте любой вопрос о программах, курсах или поступлении.',
            'keyboard': [["🔙 Назад в главное меню"]]
        }
    
    def handle_smart_mode(self, user_id: int, username: str) -> Dict:
        """Обработка команды /smart"""
        self.user_states[user_id] = {'state': 'smart_mode'}
        return {
            'text': '🧠 Вы перешли в локальную умную систему!\n\nТеперь я буду отвечать на ваши вопросы, используя базу знаний ИТМО.\n\nЗадайте любой вопрос о программах, курсах или поступлении.',
            'keyboard': [["🔙 Назад в главное меню"]]
        }
    
    def handle_gpt_question(self, user_id: int, username: str, question: str) -> Dict:
        """Обработка вопроса в GPT режиме"""
        if not self.gpt_available:
            return {
                'text': 'GPT режим временно недоступен.',
                'keyboard': self.get_main_keyboard()
            }
        
        try:
            # Получаем контекст пользователя
            user = self.db.get_user_by_telegram_id(user_id)
            user_context = None
            if user:
                user_context = {
                    'interests': user.get('interests', []),
                    'background': user.get('background', {}),
                    'preferred_program': user.get('preferred_program')
                }
            
            # Генерируем ответ через GPT
            result = self.gpt.generate_smart_answer(question, user_context)
            
            response_text = result['answer']
            response_text += f"\n\n🤖 Умный ответ на основе базы знаний ИТМО"
            
            return {
                'text': response_text,
                'keyboard': self.get_gpt_keyboard()
            }
            
        except Exception as e:
            logger.error(f"Ошибка в GPT режиме: {e}")
            return {
                'text': 'Произошла ошибка. Попробуйте позже.',
                'keyboard': self.get_main_keyboard()
            }
    
    def handle_smart_question(self, user_id: int, username: str, question: str) -> Dict:
        """Обработка вопроса в локальном умном режиме"""
        try:
            # Получаем контекст пользователя
            user = self.db.get_user_by_telegram_id(user_id)
            user_context = None
            if user:
                user_context = {
                    'interests': user.get('interests', []),
                    'background': user.get('background', {}),
                    'preferred_program': user.get('preferred_program')
                }
            
            # Генерируем ответ через локальную умную систему
            result = self.smart_qa.generate_smart_answer(question, user_context)
            
            response_text = result['answer']
            response_text += f"\n\n🧠 Локальный умный анализ ИТМО"
            
            return {
                'text': response_text,
                'keyboard': self.get_smart_keyboard()
            }
            
        except Exception as e:
            logger.error(f"Ошибка в локальном умном режиме: {e}")
            return {
                'text': 'Произошла ошибка. Попробуйте позже.',
                'keyboard': self.get_main_keyboard()
            }
    
    def handle_program_comparison_gpt(self, user_id: int, message: str) -> Dict:
        """Обработка сравнения программ через GPT"""
        if not self.gpt_available:
            self.user_states.pop(user_id, None)
            return {
                'text': 'GPT режим временно недоступен.',
                'keyboard': self.get_main_keyboard()
            }
        
        try:
            # Сбрасываем состояние
            self.user_states.pop(user_id, None)
            
            # Получаем сравнение программ через GPT
            result = self.gpt.get_program_comparison()
            
            if result.get('success'):
                response_text = result['comparison']
                response_text += f"\n\n🤖 Детальное сравнение от ИИ-консультанта"
                
                return {
                    'text': response_text,
                    'keyboard': self.get_main_keyboard()
                }
            else:
                return {
                    'text': f"Ошибка при сравнении программ: {result.get('error', 'Неизвестная ошибка')}",
                    'keyboard': self.get_main_keyboard()
                }
                
        except Exception as e:
            logger.error(f"Ошибка при сравнении программ через GPT: {e}")
            self.user_states.pop(user_id, None)
            return {
                'text': 'Произошла ошибка при сравнении программ.',
                'keyboard': self.get_main_keyboard()
            }
    
    def handle_program_comparison(self, user_id: int, username: str) -> Dict:
        """Обработка команды сравнения программ"""
        if not self.gpt_available:
            return {
                'text': 'GPT режим временно недоступен. Используйте /programs для просмотра информации о программах.',
                'keyboard': self.get_main_keyboard()
            }
        
        try:
            # Сразу получаем сравнение через GPT
            result = self.gpt.get_program_comparison()
            
            if result.get('success'):
                response_text = result['comparison']
                response_text += f"\n\n🤖 Подробное сравнение от ИИ-консультанта"
                
                return {
                    'text': response_text,
                    'keyboard': self.get_main_keyboard()
                }
            else:
                return {
                    'text': f"Ошибка при сравнении программ: {result.get('error', 'Неизвестная ошибка')}",
                    'keyboard': self.get_main_keyboard()
                }
                
        except Exception as e:
            logger.error(f"Ошибка при сравнении программ: {e}")
            return {
                'text': 'Произошла ошибка при сравнении программ.',
                'keyboard': self.get_main_keyboard()
            }
    
    # Клавиатуры
    def get_main_keyboard(self) -> List[List[str]]:
        """Основная клавиатура"""
        keyboard = [
            ["📚 Какие программы доступны?", "⏱️ Сколько длится обучение?"],
            ["🎯 Требования для поступления?", "💼 Карьерные перспективы?"],
            ["🔍 Получить рекомендации", "📖 Посмотреть курсы"],
            ["❓ Задать вопрос", "⚙️ Настроить профиль"]
        ]
        
        # Добавляем умные функции
        if self.gpt_available:
            keyboard.append(["🤖 Умный ответ", "📊 Сравнить программы"])
        else:
            keyboard.append(["🧠 Умный ответ", "📊 Сравнить программы"])
        
        return keyboard
    
    def get_programs_keyboard(self) -> List[List[str]]:
        """Клавиатура для работы с программами"""
        keyboard = [
            ["🔍 Получить рекомендации", "📖 Посмотреть курсы"],
            ["❓ Задать вопрос", "🔙 Главное меню"]
        ]
        
        keyboard.insert(1, ["📊 Сравнить программы"])
        
        return keyboard
    
    def get_gpt_keyboard(self) -> List[List[str]]:
        """Клавиатура для внешнего GPT режима"""
        return [
            ["🤖 Умный ответ", "📊 Сравнить программы"],
            ["🔍 Получить рекомендации", "🔙 Главное меню"]
        ]
    
    def get_smart_keyboard(self) -> List[List[str]]:
        """Клавиатура для локального умного режима"""
        return [
            ["🧠 Умный ответ", "📊 Сравнить программы"],
            ["🔍 Получить рекомендации", "🔙 Главное меню"]
        ]

# Пример использования
if __name__ == "__main__":
    bot = BotHandler()
    
    # Тестируем основные сценарии
    test_messages = [
        ("/start", "Команда старт"),
        ("📚 Какие программы доступны?", "Быстрый вопрос о программах"),
        ("Сколько длится обучение?", "Свободный вопрос"),
        ("/recommend", "Запрос рекомендаций")
    ]
    
    print("Тестирование BotHandler:")
    print("=" * 50)
    
    for message, description in test_messages:
        print(f"\n{description}: '{message}'")
        response = bot.process_message(12345, "test_user", message)
        print("Ответ:")
        print(response['text'][:200] + "..." if len(response['text']) > 200 else response['text'])
        print(f"Клавиатура: {len(response.get('keyboard', []))} кнопок")
        print("-" * 30) 