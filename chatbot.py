import warnings
import wikipedia
import spacy
from transformers import pipeline
import requests

# Suppress specific warnings about truncation
warnings.filterwarnings("ignore", category=UserWarning, message=".*Truncation.*")
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Use GPT-Neo (or GPT-J) from HuggingFace
chat_generator = pipeline("text-generation", model="gpt2")

# Function to get facts from Open Trivia Database
def get_trivia_facts():
    url = "https://opentdb.com/api.php?amount=1&type=multiple"
    response = requests.get(url)
    data = response.json()
    
    # Return trivia question and correct answer
    if data['results']:
        trivia = data['results'][0]
        question = trivia['question']
        correct_answer = trivia['correct_answer']
        return f"Trivia: {question}\nAnswer: {correct_answer}"
    else:
        return "Sorry, I couldn't fetch trivia at the moment."

# Predefined general questions and answers
general_questions = {
    "what is ai?": "AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn.",
    "what is machine learning?": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
    "what is deep learning?": "Deep Learning is a subset of Machine Learning involving neural networks with multiple layers, allowing machines to process complex patterns.",
    "what is your name?": "I'm your chatbot assistant. You can call me ChatBot.",
    "how are you?": "I'm just a bot, but I'm functioning perfectly! How can I assist you today?",
    "can you help me?": "Of course! Let me know what you need help with, and I'll do my best to assist you.",
    "what is python?": "Python is a versatile, high-level programming language known for its readability and broad range of applications.",
    "tell me a joke": "Why don’t programmers like nature? It has too many bugs!",
    "what is the capital of france?": "The capital of France is Paris.",
    "who is the current president of the united states?": "As of 2024, the current President of the United States is Joe Biden.",
    "what is your purpose?": "My purpose is to assist you with your queries and provide helpful information!",
    "what is your favorite color?": "I don't have a preference, but I think all colors are beautiful!",
    "what is your favorite food?": "I don't eat, but I enjoy helping humans discuss their favorite meals!",
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help?",
    "hey": "Hey! What's on your mind?",
    "goodbye": "Goodbye! Have a wonderful day!",
    "bye": "Bye! Take care!",
    "thank you": "You're welcome! Let me know if you need further assistance."
}

# Function to generate responses from Wikipedia or GPT-Neo/GPT-J
def generate_response(user_input):
    # Handle specific hardcoded questions
    user_input_lower = user_input.lower()
    if user_input_lower in general_questions:
        return general_questions[user_input_lower]

    # Handle predefined questions that need slight variation
    if user_input_lower == "what is nlp?":
        return "NLP stands for Natural Language Processing. It is the ability of a machine to understand and respond to human language."

    if user_input_lower == "who are you?":
        return "I'm your chatbot assistant, here to help you with your queries!"

    try:
        # Check for Wikipedia response first
        search_results = wikipedia.search(user_input, results=3)

        # If no relevant Wikipedia page is found, fallback to GPT-Neo
        if not search_results:
            return generate_gpt_response(user_input)

        # If Wikipedia has a result, fetch the summary
        for result in search_results:
            try:
                summary = wikipedia.summary(result, sentences=2)
                if user_input_lower in summary.lower():  # Ensure relevance
                    return summary
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Sorry, there are multiple meanings for '{user_input}'. Can you be more specific? Options: {e.options[:5]}"
            except wikipedia.exceptions.PageError:
                return "Sorry, I couldn't find any relevant information on Wikipedia."
            except wikipedia.exceptions.RedirectError:
                return "Sorry, the page you're looking for redirects to another page."

        # Fallback to GPT-Neo for any queries where Wikipedia doesn't give good results
        return generate_gpt_response(user_input)

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Sorry, there are multiple meanings for '{user_input}'. Can you be more specific? Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find any relevant information on Wikipedia."
    except wikipedia.exceptions.RedirectError:
        return "Sorry, I couldn't find any relevant information on Wikipedia."

# Function to generate response from GPT-Neo
def generate_gpt_response(user_input):
    prompt = (
        f"The following is a conversation with an AI assistant. "
        f"The assistant is helpful, creative, and knowledgeable about NLP and AI.\n\n"
        f"User: {user_input}\nAI:"
    )
    response = chat_generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
    return response[0]['generated_text'].split("AI:")[-1].strip()

# Function to process entities using spaCy for smarter responses
def process_entities(user_input):
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
