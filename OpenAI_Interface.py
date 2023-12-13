import json
import openai
from dotenv import load_dotenv
import tiktoken

class OpenAI_Interface:


    def __init__(self, model="gpt-4-1106-preview", temperature=0, stream=True, db_upsert_queue=None, jarvis_messages_queue=None):
        load_dotenv(".env")
        self.client = openai.OpenAI()
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.instructions = self.load_instructions("system_files/Assistant_Instructions.json")
        self.tools = [self.instructions["Functions"][function] for function in self.instructions["Functions"]]
        self.default_messages = [{"role": "system", "content": json.dumps(self.instructions["Instructions"])}]
        self.messages = self.load_messages("system_files/messages.json", "Alec")
        self.responding = False
        self.max_tokens = 128000
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.tokens_in_context = (self.count_tokens(json.dumps(self.instructions["Instructions"])))
        self.db_upsert_queue = db_upsert_queue
        self.jarvis_messages_queue = jarvis_messages_queue


    def load_instructions(self, filename):
        with open(filename, 'r') as file:
            instructions = json.load(file)
        return instructions


    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    

    def save_messages(self, file_path, user_name):
            try:
                data = {}
                with open(file_path, 'r') as file:
                    data = json.load(file)
                data[user_name] = self.messages[1:]

                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
            
            except FileNotFoundError:
                print(f"No existing file found. A new file will be created at {file_path}")
            except json.JSONDecodeError:
                print(f"Existing file at {file_path} is not a valid JSON. It will be overwritten.")
            except Exception as e:
                print(f"Error saving messages for {user_name} to JSON: {e}")


    def load_messages(self, file_path, user_name):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                user_messages = data.get(user_name)
                if user_messages is not None:
                    return self.default_messages + user_messages
                else:
                    return self.default_messages
        except FileNotFoundError:
            print(f"File {file_path} not found. Initializing with default messages.")
            return self.default_messages
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}. Initializing with default messages.")
            return self.default_messages
        except Exception as e:
            print(f"Error loading messages for {user_name} from JSON: {e}")
            return self.default_messages


    def summarize_and_reset_context(self):
        summary_instructions = json.dumps(self.instructions["Write_SPR"])
        conversation_history = "\n".join([msg['content'] for msg in self.messages])
        summary_message = f"{summary_instructions}\n{conversation_history}"

        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": summary_message}],
            temperature=self.temperature
        ).choices[0].message

        self.db_upsert_queue.put(summary_response)
        print("Summary response added to db_write queue.\n")

        self.messages = [] + self.default_messages
        self.tokens_in_context = self.count_tokens(json.dumps(self.instructions["Instructions"]))


    def message(self, role, content, tools):

        new_msg_token_count = self.count_tokens(content)
        if self.tokens_in_context + new_msg_token_count + self.count_tokens(json.dumps(self.instructions["Write_SPR"])) >= self.max_tokens:
            self.summarize_and_reset_context()

        try:
            self.messages.append({
                "role": role,
                "content": content,
            })
            self.tokens_in_context += new_msg_token_count

            if role == "user":
                self.responding = True
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    tools=tools,
                    stream=self.stream,
                )

                func_call = {"name": None, "arguments": ""}
                collecting_tokens = False
                current_sentence = ""
                system_message = {"role": "system", "content": ""}

                for chunk in response:
                    tool_calls = chunk.choices[0].delta.tool_calls if chunk.choices[0].delta and chunk.choices[0].delta.tool_calls else None
                    if not tool_calls:
                        continue

                    for func_chunk in tool_calls:
                        function = func_chunk.function if func_chunk else None

                        if not function:
                            continue

                        if function.name:
                            if func_call["name"] is not None:
                                system_message["content"] += json.dumps(func_call)
                                self.process_function_call(func_call)
                            func_call["name"] = function.name
                            func_call["arguments"] = ""
                            continue

                        if not function.arguments:
                            continue

                        if func_call["name"] == "respond_to_user":
                            if collecting_tokens:
                                current_sentence += function.arguments
                                if any(punctuation in current_sentence for punctuation in ('.', '?', '!')):
                                    current_sentence.strip('"')
                                    self.jarvis_messages_queue.put(current_sentence)
                                    current_sentence = ""
                            if function.arguments == '":"':
                                collecting_tokens = True
                            continue

                        func_call["arguments"] += function.arguments

                if func_call["name"] is not None:
                    system_message["content"] += json.dumps(func_call)
                    self.process_function_call(func_call)

                self.tokens_in_context += self.count_tokens(system_message["content"])
                self.messages.append(system_message)

        except Exception as e:
            print(f"MESSAGE OPENAI FAILED: {e}")
            self.responding = False


    def process_function_call(self, func_call):
        function_name = func_call["name"]
        arguments = func_call["arguments"]
        arguments = arguments.strip('}{')
        start_index = arguments.find(':') + 1
        arguments = arguments[start_index:]

        if function_name == "save_to_vector_database":
            self.db_upsert_queue.put(arguments)


    def make_prompt(self, user, user_emotion, timestamp, text, query_results):
        formatted_query_results = "\n\n".join(f"{index + 1}. {result}" for index, result in enumerate(query_results))

        if query_results:
            prompt =("**Metadata**\n"
                    f"User: {user} \n"
                    f"User emotion: {user_emotion} \n"
                    f"Timestamp: {timestamp}\n\n"

                    "**User's message**\n"
                    f"{text}\n\n"

                    "**Vector database results**\n"
                    f"{formatted_query_results}\n\n"

                    "**Response guidelines**\n"
                    "Format your response as a response call first, and then include as many database calls you deem necessary. "
                    "Your response should remain within the context of the user's message and avoid tangential topics. "
                    "Use the metadata to augment your response if it's relevant. "
                    "Ie. acknowledge significant emotions, demonstrate temporal awareness, etc. "
                    "Use the most relevant information from the vector database results to enrich and personalize your response. "
                    "Do not save information from the vector database results to the vector database to avoid redundancy."
                    )
        else:
            prompt =("**Metadata**\n"
                    f"User: {user} \n"
                    f"User emotion: {user_emotion} \n"
                    f"Timestamp: {timestamp}\n\n"

                    "**User's essage to Jarvis**\n"
                    f"{text}\n\n"

                    "**Response guidelines**\n"
                    "Format your response as a response call first, and then include as many database calls you deem necessary. "
                    "Your response should remain within the context of the user's message and avoid tangential topics. "
                    "Use the metadata to augment your response if it's relevant. "
                    "Ie. acknowledge significant emotions, demonstrate temporal awareness, etc. "
                    )
        
        return prompt