import logging
from openai import OpenAI
from termcolor import colored  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = "Bielik-11B-v2.5-Instruct" # Replace with your desired model
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1") # Adjust if needed
logging.info(f"Using model: {model}")

role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "blue",
    "reasoning": "yellow"
}

def pretty_print_conversation(messages):
    for message in messages:
        role = message["role"]
        base_color = role_to_color.get(role, "white")
        print(colored(f"{role}: ", base_color), end="")

        if role == "system":
            print(colored(message.get('content', '[No content]'), base_color))
        elif role == "user":
            print(colored(message.get('content', '[No content]'), base_color))
        elif role == "assistant":
            parts_to_print = []
            if message.get("reasoning") is not None:
                parts_to_print.append(colored("[ Thinking... ]\n" + message['reasoning'] + "\n[ Thinking finished ]", role_to_color.get("reasoning", "white")))
            
            if message.get("content") is not None:
                parts_to_print.append(colored(message['content'], base_color))
            
            if not parts_to_print:
                print(colored("[No output from assistant]", base_color))
            else:
                print("".join(parts_to_print))
        else:
            print(colored(str(message), base_color))
        
        print() # Add a newline after each message for separation

def chat_completion_request(messages, enable_thinking):
    try:
        if enable_thinking:
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0, # the high temperature to generate more creative and varied reasoning paths
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
            )
        else:
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000, # prevent long outputs
                temperature=0.2,
                stream=True
            )
        return response_stream
    except Exception as e:
        logging.warning(f"Unable to generate ChatCompletion response. Exception: {e}")
        return e

def process_streamed_response(stream, print_stream=False):
    full_response_content = ""
    full_reasoning_content = ""
    reasoning = False
    assistant_color = role_to_color.get("assistant", "white")
    reasoning_color = role_to_color.get("reasoning", "white")

    for chunk in stream:
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            full_reasoning_content += delta.reasoning_content
            if print_stream:
                if not reasoning:
                    print(colored("[ Thinking... ]\n", reasoning_color), end="", flush=True)
                print(colored(delta.reasoning_content, reasoning_color), end="", flush=True)
            reasoning = True
        elif hasattr(delta, "content") and delta.content:
            full_response_content += delta.content
            if print_stream:
                if reasoning:
                    print(colored("[ Thinking finished ]\n", reasoning_color), end="", flush=True)
                    reasoning = False
                print(colored(delta.content, assistant_color), end="", flush=True)
        
        if finish_reason:
            break # Exit loop once a finish reason is received

    if full_reasoning_content and not full_response_content:
        # workaround to handle cases where only reasoning is provided
        full_response_content = full_reasoning_content
        full_reasoning_content = ""
        if print_stream:
            if reasoning:
                print(colored("\n[ Thinking finished ]\n", reasoning_color), end="", flush=True)
                reasoning = False
            print(colored(full_response_content, assistant_color), end="", flush=True)
    
    if print_stream:
        print()

    assistant_message_dict = {"role": "assistant", "content": None, "reasoning": None}
    
    if full_reasoning_content:
        assistant_message_dict["reasoning"] = full_reasoning_content
    if full_response_content:
        assistant_message_dict["content"] = full_response_content
    else:
        assistant_message_dict["content"] = "" 

    return assistant_message_dict

def add_turn(prompt, messages, enable_thinking=False):
    messages.append({"role": "user", "content": prompt})

    stream = chat_completion_request(messages, enable_thinking)
    if isinstance(stream, Exception):
        logging.error(f"Error in first API call: {stream}")
        # Add an error message to conversation history for the assistant's turn
        messages.append({"role": "assistant", "content": f"API Error: Could not get response. {stream}"})
        print(colored(f"assistant: API Error: Could not get response. {stream}", "red"))
        return

    print(colored(f"assistant: ", role_to_color.get("assistant")), end="", flush=True)
    assistant_response_dict = process_streamed_response(stream, print_stream=True)
    messages.append(assistant_response_dict)


if __name__ == "__main__":
    messages = []
    # Optional: Add a system prompt
    # messages.append({"role": "system", "content": "Jesteś pomocnym asystentem, który pomaga w podróży."})

    prompts = [
        "Wymyśl i napisz mi krótkie motywujące zdanie na dziś",
        "Jade na jednodniową wycieczkę do Końskich. Co warto zobaczyć?",
        "To teraz krótki motywujacy tekst na ten temat",
    ]

    for i, prompt in enumerate(prompts):
        with_thinking = True if i == 1 else False
        print(colored(f"\n--- Turn {i+1} ---", "yellow", attrs=["bold"]))
        print(colored(f"user: {prompt}", role_to_color.get("user")))
        add_turn(prompt, messages, enable_thinking=with_thinking)
    
    logging.info(f"--- Final Conversation History ---")
    pretty_print_conversation(messages)
