from transformers import AutoTokenizer

# Initialize Tokenizer
HF_TOKEN='<insert_token>'
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', token=HF_TOKEN)

# Load the modified chat template to support message + tool call features
tokenizer.chat_template = '''
{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
{%- set ns = namespace() %}
{%- set ns.index = 0 %}
{%- for message in loop_messages %}
    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if message["role"] == "user" %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
            {%- endif %}
        {%- if loop.last and system_message is defined %}
            {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
        {%- else %}
            {{- "[INST] " + message["content"] + "[/INST]" }}
        {%- endif %}
    {%- elif message.tool_calls is defined and message.tool_calls is not none %}
        {{- message.content if 'content' in message}}
        {{- "[TOOL_CALLS] [" }}
        {%- for tool_call in message.tool_calls %}
            {%- set out = tool_call.function|tojson %}
            {{- out[:-1] }}
            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
            {%- endif %}
            {{- ', "id": "' + tool_call.id + '"}' }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" + eos_token }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["role"] == "assistant" %}
        {{- " " + message["content"]|trim + eos_token}}
    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS] {"content": ' + content|string + ", " }}
        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
        {%- endif %}
        {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}
'''

# Define a tool:
get_stock_price = {"type": "function", "function": {
    "name": "get_stock_price", 
    "description": "Get the current stock price of a company", 
    "parameters": {
        "type": "object", 
        "properties": {"company": {"type": "string", "description": "The name of the company"}}, 
    "required": ["company"]}}}

example = {
    "messages": [
        {"role": "system", "content": "Use the functions if necessary. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous. Think before you respond."}, 
        {"role": "user", "content": "What is the stock price of Apple?"},
        {"role": "assistant", "content": f'[THOUGHT] To get the current stock price of Apple, we can use the provided function `{get_stock_price["function"]["name"]}(company="Apple")`. [/THOUGHT] ', "tool_calls": [{"id": "O3MwVtQGm", "type": "function", "function": {"name": get_stock_price["function"]["name"], "arguments": "{'company': 'Apple'}"}}]},
        {"role": "tool", "content": "220.75}", "tool_call_id": "O3MwVtQGm"}, 
        {"role": "assistant", "content": "The current stock price of Apple is $220.75."}, 
        {"role": "user", "content": "What about Starbucks?"}, 
    ],
    "tools": [
        get_stock_price,
    ]
}

# To format the conversation for inference, we can apply the chat template
#   and decode the subsequent tokens
tokenized_message = tokenizer.apply_chat_template(conversation=example['messages'], tools=example['tools'])
formatted_prompt = tokenizer.decode(tokenized_message)
print(formatted_prompt)