from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import chainlit as cl

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check API Key
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Define the Agent
agent = Agent(
    name="Client Communication Coach",
    instructions="""
You are a professional freelancer mentor and communication expert.

Your task is to help users respond to difficult clients with calm, professional, and polite language. 
You should never be rude or defensive. Maintain empathy, confidence, and clarity in your tone.

If the user gives you a rough, emotional, or unclear message, rewrite it into a well-worded, professional reply that they can send to a client.
"""
)

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Choose the model (Gemini Flash)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Define configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Start of Chainlit session
@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="👋 Hello! I'm your Client Communication Coach. Paste the rough message you'd like to send to a client, and I’ll rewrite it professionally.").send()

# Handling incoming messages
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    # Add user input to history
    history.append({"role": "user", "content": message.content})

    # Send an empty message and prepare to stream output
    msg = cl.Message(content="")
    await msg.send()

    # Stream the AI response
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Save assistant response in history
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
