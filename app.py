from dotenv import load_dotenv
import chainlit as cl
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket, get_reviews
import json
import movie_functions
import inspect
import asyncio

load_dotenv()
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a movie guide bot. For each query, decide whether to use your knowledge base or fetch context using specific methods. Follow these guidelines:

1. Use Knowledge Base for:
   - General movie facts, trivia, recommendations, and summaries.
   - Known information on actors, genres, awards, or classic films.

2. Fetch Context with:
   - get_now_playing_movies(): For currently showing films.
   - get_showtimes(): For movie times at specific locations.
   - buy_ticket(): To assist with ticket purchases.
   - get_reviews(): For recent reviews or audience reactions.
   - confirm_ticket_purchase(): Confirm with the user, before making the purchase of the ticket.

3. Interaction: Be clear and concise. Ask for clarification if needed. Keep a friendly and helpful tone.
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    while True:
        functions = [
            {
                "name": "get_now_playing_movies",
                "description": "Get a list of movies currently playing in theaters"
            },
            {
                "name": "get_showtimes",
                "description": "Get showtimes for a specific movie at a given location"
            },
            {
                "name": "buy_ticket",
                "description": "Purchase a ticket for a specific movie showing"
            },
            {
                "name": "get_reviews",
                "description": "Get recent reviews for a specific movie"
            },
            {
                "name":"confirm_ticket_purchase",
                "description":"Confirm the details with the user before purchasing the ticket"
            }
        ]
        
        try:
            function_call_response = await client.chat.completions.create(
                model="gpt-4",
                messages=message_history,
                functions=functions,
                function_call="auto"
            )
            
            response_choice = function_call_response.choices[0]

            if not response_choice.message.function_call:
                break

            function_name = response_choice.message.function_call.name
            function_args = json.loads(response_choice.message.function_call.arguments)
            
            if hasattr(movie_functions, function_name):
                function_to_call = getattr(movie_functions, function_name)
                
                expected_args = inspect.signature(function_to_call).parameters.keys()
                adjusted_args = {k: v for k, v in function_args.items() if k in expected_args}
                
                try:
                    if inspect.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**adjusted_args)
                    else:
                        function_response = function_to_call(**adjusted_args)

                    message_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_response
                    })
                except Exception as e:
                    error_message = f"Error calling function {function_name}: {str(e)}"
                    message_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": error_message
                    })
            else:
                error_message = f"Function {function_name} not found"
                message_history.append({
                    "role": "function",
                    "name": function_name,
                    "content": error_message
                })
        except Exception as e:
            error_message = f"Error in function call loop: {str(e)}"
            message_history.append({
                "role": "system",
                "content": error_message
            })
            break

    response_message = await generate_response(client, message_history, gen_kwargs)

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()