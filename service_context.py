from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from langchain.chat_models import ChatOpenAI


def load_service_context():
    try:
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
                temperature=0.3, model_name="gpt-3.5-turbo", max_tokens=num_outputs
            )
        )

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )

        return service_context

    except Exception as e:
        print("Error loading service context", e)
        return (
            f"Error loading service context. Please inform your developer. Error: {e}",
            "",
        )
