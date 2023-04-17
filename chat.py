from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
import logging
import sys
from dotenv import load_dotenv
import os.path
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables from .env file
load_dotenv()
# Get the value of OPENAI_API_KEY from the environment
api_key = os.getenv("OPENAI_API_KEY")
# Use the API key in your code
os.environ["OPENAI_API_KEY"] = api_key

# Defining the parameters for the index
max_input_size = 4096
num_outputs = 1024
max_chunk_overlap = 20

prompt_helper = PromptHelper(
    max_input_size,
    num_outputs,
    max_chunk_overlap,
)

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", max_tokens=num_outputs)
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

index = GPTSimpleVectorIndex.load_from_disk(
    "index.json", service_context=service_context
)

# Optimiser reduced time and token usage
# Tools for langchain agent
tools = [
    Tool(
        name="Custom GPT Index",
        func=lambda q: str(
            index.query(q, optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5))
        ),
        description="useful for when you want to answer questions about projects "
        "and information from the documents. "
        "Input to this should be tell me about or summarise something",
        return_direct=True,  # return the direct response (observation) from the tool
    ),
]

# set Logging to DEBUG for more detailed outputs
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

while True:
    user_input = input("\nQuestion: ")
    if user_input.lower() == "quit":
        break
    response = agent_chain.run(input=user_input)
    print("AI:", response)
