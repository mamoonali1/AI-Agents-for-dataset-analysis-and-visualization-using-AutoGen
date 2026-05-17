from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_ext.models.openai import OpenAICompletionModel
from autogen_ext.code_executor.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
import asyncio

async def main():
    model= OpenAICompletionModel(
        model="o3-mini",
        api_key=open("key.txt").read().strip()
    )

    developer= AssistantAgent(
        name='Developer',
        model_client=model,
        system_message=(
        """You are a helpful assistant for software developers agent.
        You will be given a csv file and a question about the data in the csv file. 
        You need to analyze the csv file and answer the question in python code based on the data in the csv file.
        You should always begin with your plan to answer the question, and then write the code in a code block with language python specified.
        If you need several code blocks to answer the question, you can write one code blocks at a time.
        once you have the code exceution result, you should provifde the final result
        after that you should exactly say "Terminate" to end the conversation."""
        ),
    )

    docker= DockerCommandLineCodeExecutor(
        working_dir="temp",
    )

    executor= CodeExecutorAgent(
        name='CodeExecutor',
        code_executor=docker,

    )

    team= RoundRobinGroupChat(
        participants=[developer, executor],
        termination_condition=TextMentionTermination("Terminate"),
        max_turns=10,
    )