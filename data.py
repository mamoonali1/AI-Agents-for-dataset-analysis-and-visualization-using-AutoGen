from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_ext.models.openai import OpenAICompletionModel
from autogen_ext.code_executor.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult
import asyncio

async def teamConfig():
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
        Used pandas to analyze the csv file and answer the question.
        If library si not installed in the docker, you can install it by pip install in the code block.
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
    return team, docker


async def orchestrate(team, docker, task):
    await docker.start()
    async for message in team.run_stream():
        if isinstance(message, TextMessage):
            print(message:=f"{message.source}: {message.content}")
            yield message
        elif isinstance(message, TaskResult):
            print(message:=f"Stop reason: {message.stop_reason}")
            yield message

    await docker.stop()

async def main():
        task="My dataset is Data.csv, How many columns are in it?"
        team, docker= await teamConfig()
        async for message in orchestrate(team, docker, task):
            pass

if __name__ == "__main__":
    asyncio.run(main())


       