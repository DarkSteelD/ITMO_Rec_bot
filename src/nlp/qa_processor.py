import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from src.database.db_manager import DatabaseManager
from config import SIMILARITY_THRESHOLD, MIN_RELEVANCE_SCORE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAProcessor:
    def __init__(self, db_manager: DatabaseManager = None):
        """Инициализация процессора вопросов и ответов"""
        self.db = db_manager or DatabaseManager()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None  # Будем обрабатывать русские стоп-слова сами
        )
        
        # Инициализация NLTK компонентов
        self._init_nltk()
        
        # Загружаем данные
        self.qa_pairs = self.db.get_all_qa_pairs()
        self.questions = [qa['question'] for qa in self.qa_pairs]
        self.answers = [qa['answer'] for qa in self.qa_pairs]
        
        # Предобрабатываем вопросы и создаем векторы
        self.processed_questions = [self._preprocess_text(q) for q in self.questions]
        
        if self.processed_questions:
            self.question_vectors = self._fit_vectorizer()
        else:
            self.question_vectors = None
            logger.warning("Нет данных для обучения векторизатора")
    
    def _init_nltk(self):
        """Инициализация NLTK данных"""
        try:
            # Скачиваем необходимые данные NLTK
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Инициализируем стеммер и стоп-слова
            self.stemmer = SnowballStemmer('russian')
            self.stop_words = set(stopwords.words('russian'))
            
            # Добавляем дополнительные стоп-слова
            additional_stops = {
                'это', 'быть', 'мочь', 'весь', 'свой', 'который', 'такой',
                'только', 'один', 'время', 'год', 'человек', 'сказать'
            }
            self.stop_words.update(additional_stops)
            
        except Exception as e:
            logger.error(f"Ошибка инициализации NLTK: {e}")
            # Fallback к базовому набору
            self.stemmer = None
            self.stop_words = set()
    
    def _preprocess_text(self, text: str) -> str:
        """Предобработка текста"""
        if not text:
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем лишние символы
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Токенизация
        try:
            tokens = word_tokenize(text, language='russian')
        except:
            tokens = text.split()
        
        # Удаляем стоп-слова и стеммируем
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.stemmer:
                    try:
                        stemmed = self.stemmer.stem(token)
                        processed_tokens.append(stemmed)
                    except:
                        processed_tokens.append(token)
                else:
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def _fit_vectorizer(self) -> np.ndarray:
        """Обучение векторизатора на имеющихся вопросах"""
        try:
            vectors = self.vectorizer.fit_transform(self.processed_questions)
            return vectors.toarray()
        except Exception as e:
            logger.error(f"Ошибка при создании векторов: {e}")
            return None
    
    def find_similar_question(self, user_question: str) -> Tuple[Optional[Dict], float]:
        """Поиск наиболее похожего вопроса"""
        if not self.question_vectors is not None:
            return None, 0.0
        
        # Предобрабатываем пользовательский вопрос
        processed_question = self._preprocess_text(user_question)
        
        try:
            # Векторизуем вопрос пользователя
            user_vector = self.vectorizer.transform([processed_question]).toarray()
            
            # Вычисляем косинусное сходство
            similarities = cosine_similarity(user_vector, self.question_vectors)[0]
            
            # Находим наиболее похожий вопрос
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity >= MIN_RELEVANCE_SCORE:
                return self.qa_pairs[best_match_idx], best_similarity
            else:
                return None, best_similarity
                
        except Exception as e:
            logger.error(f"Ошибка при поиске похожего вопроса: {e}")
            return None, 0.0
    
    def get_answer(self, user_question: str) -> Dict:
        """Получение ответа на вопрос пользователя"""
        # Ищем наиболее похожий вопрос
        similar_qa, similarity = self.find_similar_question(user_question)
        
        if similar_qa and similarity >= SIMILARITY_THRESHOLD:
            return {
                'answer': similar_qa['answer'],
                'confidence': similarity,
                'matched_question': similar_qa['question'],
                'category': similar_qa.get('category', 'general'),
                'is_exact_match': similarity > 0.9
            }
        elif similar_qa:
            return {
                'answer': f"Возможно, вы имели в виду: '{similar_qa['question']}'?\n\n{similar_qa['answer']}",
                'confidence': similarity,
                'matched_question': similar_qa['question'],
                'category': similar_qa.get('category', 'general'),
                'is_exact_match': False
            }
        else:
            return {
                'answer': "К сожалению, я не могу найти ответ на ваш вопрос. Попробуйте переформулировать вопрос или обратитесь к администратору программы.",
                'confidence': 0.0,
                'matched_question': None,
                'category': 'unknown',
                'is_exact_match': False
            }
    
    def get_related_questions(self, user_question: str, top_k: int = 3) -> List[Dict]:
        """Получение связанных вопросов"""
        if not self.question_vectors is not None:
            return []
        
        processed_question = self._preprocess_text(user_question)
        
        try:
            user_vector = self.vectorizer.transform([processed_question]).toarray()
            similarities = cosine_similarity(user_vector, self.question_vectors)[0]
            
            # Получаем индексы top_k наиболее похожих вопросов
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            related = []
            for idx in top_indices:
                if similarities[idx] >= MIN_RELEVANCE_SCORE:
                    related.append({
                        'question': self.qa_pairs[idx]['question'],
                        'answer': self.qa_pairs[idx]['answer'],
                        'similarity': similarities[idx],
                        'category': self.qa_pairs[idx].get('category', 'general')
                    })
            
            return related
            
        except Exception as e:
            logger.error(f"Ошибка при поиске связанных вопросов: {e}")
            return []
    
    def add_qa_pair(self, question: str, answer: str, category: str = 'general') -> bool:
        """Добавление новой пары вопрос-ответ"""
        try:
            # Извлекаем ключевые слова из вопроса
            keywords = self._extract_keywords(question)
            
            # Добавляем в базу данных
            self.db.insert_qa_pair(question, answer, category, keywords=keywords)
            
            # Обновляем внутренние данные
            self._reload_data()
            
            logger.info(f"Добавлена новая пара Q&A: {question[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении Q&A пары: {e}")
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        processed = self._preprocess_text(text)
        words = processed.split()
        
        # Возвращаем уникальные слова длиннее 3 символов
        keywords = list(set([word for word in words if len(word) > 3]))
        return keywords[:10]  # Ограничиваем количество ключевых слов
    
    def _reload_data(self):
        """Перезагрузка данных из базы"""
        self.qa_pairs = self.db.get_all_qa_pairs()
        self.questions = [qa['question'] for qa in self.qa_pairs]
        self.answers = [qa['answer'] for qa in self.qa_pairs]
        
        self.processed_questions = [self._preprocess_text(q) for q in self.questions]
        
        if self.processed_questions:
            self.question_vectors = self._fit_vectorizer()
    
    def get_statistics(self) -> Dict:
        """Получение статистики по Q&A базе"""
        categories = {}
        for qa in self.qa_pairs:
            category = qa.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_qa_pairs': len(self.qa_pairs),
            'categories': categories,
            'avg_question_length': np.mean([len(q) for q in self.questions]) if self.questions else 0,
            'avg_answer_length': np.mean([len(a) for a in self.answers]) if self.answers else 0
        }

# Пример использования
if __name__ == "__main__":
    # Инициализируем процессор
    processor = QAProcessor()
    
    # Тестовые вопросы
    test_questions = [
        "Сколько длится обучение?",
        "Какие требования для поступления в магистратуру?",
        "В чем отличие между программами?",
        "Где можно работать после окончания?",
        "Какой средний балл нужен для поступления?"
    ]
    
    print("Тестирование системы Q&A:")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nВопрос: {question}")
        result = processor.get_answer(question)
        print(f"Ответ (confidence: {result['confidence']:.2f}):")
        print(result['answer'])
        print("-" * 30)
    
    # Статистика
    stats = processor.get_statistics()
    print(f"\nСтатистика Q&A базы:")
    print(f"Всего пар: {stats['total_qa_pairs']}")
    print(f"Категории: {stats['categories']}") 