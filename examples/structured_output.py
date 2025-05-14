import logging
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel
from termcolor import colored  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = "Bielik-11B-v2.5-Instruct" # Replace with your desired model
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1") # Adjust if needed
logging.info(f"Using model: {model}")

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


logging.info(f"Using model: {model}")

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue"
    }
    
    for message in messages:
        base_color = role_to_color.get(message["role"], "white")
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", base_color))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", base_color))
        elif message["role"] == "assistant":
            print(colored(f"assistant: {message['content']}\n", base_color))
        else:
            print(colored(str(message), base_color))

def chat_completion_request(messages, extra_body):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            extra_body=extra_body,
        )
        return response
    except Exception as e:
        logging.warning(f"Unable to generate ChatCompletion response. Exception: {e}")
        return e

def add_turn(prompt, messages, extra_body=None):
    messages.append({"role": "user", "content": prompt})
    chat_response = chat_completion_request(messages, extra_body)
    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message.model_dump())

if __name__ == "__main__":
    json_schema = CarDescription.model_json_schema()
    logging.info(f"Configured JSON schema: {json_schema}")

    messages = []
    add_turn("Wymyśl i napisz mi krótkie motywujące zdanie na dziś", messages)
    add_turn("Wygeneruj JSON zawierający markę, model i typ nadwozia najbardziej ikonicznego samochodu z lat 90.", messages, {"guided_json": json_schema})
    add_turn("Napisz teraz krótki motywujacy tekst biorąc pod uwagę ten samochód", messages)
    add_turn("Ja jest najlepszy samochód dla 4 osobowej rodziny w Polsce? Odpowiedz w formacie JSON podając markę, model i typ nadwozia", messages, {"guided_json": json_schema})
      
    logging.info(f"Messages:")
    pretty_print_conversation(messages)
