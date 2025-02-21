from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

model = init_chat_model("llama3.2", model_provider = "ollama")

print(model.invoke([HumanMessage(content="Hi! I'm Bob")]))
print(model.invoke([HumanMessage(content="What's my name?")]))

print(model.invoke([HumanMessage(content="Hi! I'm Bob"),
                    AIMessage(content="Hello Bob! How can I assist you today?"),
                    HumanMessage(content="What's my name?"),]))