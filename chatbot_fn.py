from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os.path
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from firebase_utils import bucket

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


def chatbot_fn(input_text):
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
        llm=ChatOpenAI(
            temperature=0.2, model_name="gpt-3.5-turbo", max_tokens=num_outputs
        )
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    try:
        # Download the index.json file from Firebase Storage
        blob = bucket.blob("gptIndices/index.json")
        index_json_data = blob.download_as_text()

        index = GPTSimpleVectorIndex.load_from_string(
            index_json_data, service_context=service_context
        )
        print("index.json has been loaded successfully.")

        # Optimiser reduced time and token usage
        # Tools for langchain agent
        tools = [
            Tool(
                name="Custom GPT Index",
                func=lambda q: str(index.query(q)),
                description="useful for when you want to answer questions about projects "
                "and information from the documents. "
                "Input to this should be tell me about or summarise something",
                return_direct=True,  # return the direct response (observation) from the tool
            ),
        ]

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
        )

        response = agent.run(input_text)

        return response

    except Exception as e:
        print("Error loading index.json:", e)
        return f"Error loading index. Inform your developer, {e}"
