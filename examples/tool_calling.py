import json
import logging
from openai import OpenAI
from termcolor import colored  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = "Bielik-11B-v2.5-Instruct" # Replace with your desired model
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1") # Adjust if needed
logging.info(f"Using model: {model}")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "num_days"]
            },
        }
    },
]
logging.info(f"Available tools: {len(tools)}")

role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "blue",
    "tool": "magenta",
}

def pretty_print_conversation(messages):
    
    for message in messages:
        base_color = role_to_color.get(message["role"], "white")
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", base_color))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", base_color))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", base_color))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", base_color))
        elif message["role"] == "tool":
            print(colored(f"tool ({message['name']}): {message['content']}\n", base_color))
        else:
            print(colored(str(message), base_color))

def chat_completion_request(messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500, # prevent long outputs
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        return response
    except Exception as e:
        logging.warning(f"Unable to generate ChatCompletion response. Exception: {e}")
        return e

def call_function(name, args):
    if name == "get_current_weather":
        return json.dumps({"temperature": "25°C", "weather": "sunny"})
    if name == "get_n_day_weather_forecast":
        return json.dumps({"forecast": [{"temperature": "21°C", "weather": "rainy"}, {"temperature": "22°C", "weather": "cloudy"}, {"temperature": "23°C", "weather": "windy and sunny"}]})
    return None

def add_turn(prompt, messages):
    messages.append({"role": "user", "content": prompt})
    chat_response = chat_completion_request(messages)
    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message.model_dump())

    tool_calls = assistant_message.tool_calls
    if tool_calls:
        tool_call_id = tool_calls[0].id
        tool_function_name = tool_calls[0].function.name
        tool_function_args = json.loads(tool_calls[0].function.arguments)
        logging.info(f"Function call {tool_function_name}(args={tool_function_args})")

        result = call_function(tool_function_name, tool_function_args)
        if not result:
            logging.warning(f"Function {tool_function_name} does not exist")
            result = "{}"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_function_name,
            "content": result,
        })
        chat_response = chat_completion_request(messages)
        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message.model_dump())

if __name__ == "__main__":
    messages = []
    # Optional: Add a system prompt
    # messages.append({"role": "system", "content": "You are a helpful assistant with access to weather tools. Please provide answers based on tool outputs when relevant."})

    prompts = [
        "Wymyśl i napisz mi krótkie motywujące zdanie na dziś",
        "A tak w ogóle to jaka dziś pogoda na dworze w Końskich?",
        "To teraz krótki motywujacy tekst biorąc pod uwagę pogodę",
        "A jaka będzie pogoda przez najbliższe 3 dni w Kielcach? Prognozę podaj w tabelce.",
        "Czy jutro w Kielcach przyda mi się parasol?",
        "A za trzy dni?"
    ]

    for i, p in enumerate(prompts):
        print(colored(f"\n--- Turn {i+1} ---", "yellow", attrs=["bold"]))
        print(colored(f"user: {p}", role_to_color.get("user")))
        add_turn(p, messages)
      
    logging.info(f"--- Final Conversation History ---")
    pretty_print_conversation(messages)
