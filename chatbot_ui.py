import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
As a highly knowledgeable health assistant, your role is to accurately interpret health queries and
provide responses using our specialized health database. Follow these directives to ensure optimal user interactions:

Precision in Answers: Respond solely with information directly relevant to the user's query from our health database.
Refrain from making assumptions or adding extraneous details.

Topic Relevance: Limit your expertise to specific health-related areas:

Symptom-based Supplement Recommendations

Allergy-specific Avoidance Guidance

Health Goal-oriented Suggestions

Additional Notes for Supplement Intake

Handling Off-topic Queries: For questions unrelated to health (e.g., general knowledge questions like "Why is the sky blue?"),
politely inform the user that the query is outside the chatbot’s scope and suggest redirecting to health-related inquiries.

Promoting Health Awareness: Craft responses that emphasize good health sense, aligning with the user's symptoms, allergies, and health goals.

Contextual Accuracy: Ensure responses are directly related to the health query, utilizing only pertinent
information from our database.

Relevance Check: If a query does not align with our health database, guide the user to refine their
question or politely decline to provide an answer.

Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and
relevance to each user query.

Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on
delivering clear, concise, and direct answers.
 history: remember previous questions thate were asked by the user.
Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "HealthBot" in responses.

One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each
sentence should be unique and contribute to the overall message without redundancy.

Health Query:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # retriever is set to fetch top 3 results
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
    """
    <style>
        .appview-container .main .block-container {{
            padding-top: {padding_top}rem;
            padding-bottom: {padding_bottom}rem;
        }}
    </style>
    """.format(
        padding_top=1, padding_bottom=1
    ),
    unsafe_allow_html=True,
)
#Header section

st.markdown(
    """
    <h3 style='text-align: left; color: #4CAF50; padding-top: 20px; border-bottom: 3px solid #4CAF50;'>
        Discover the Best Health & Supplement Recommendations 💊
    </h3>
    """,
    unsafe_allow_html=True,
)

#side bar 
side_bar_message = """
Hi! 👋 I'm here to assist you with your health queries. What would you like to explore?

Here are some areas you might be interested in:

1. **Symptom-based Supplement Recommendations** 🩺
2. **Allergy-specific Avoidance Guidance** 🚫
3. **Health Goal-oriented Suggestions** 🏃‍♂️
4. **Additional Guidance for Supplement Intake** 🍽️

Feel free to ask me anything about health and supplements!
"""


with st.sidebar:
    st.title('🤖HealthBot: Your AI Wellness Companion')
    st.markdown(side_bar_message)

#initial message
initial_message = """
    Hi there! I'm your HealthBot 🤖
    Here are some questions you might ask me:
    Here are some questions you might ask me:\n
     What supplements can help with muscle cramps?  
     Can you suggest something for improving mental focus?  
     What should I avoid if I'm allergic to gluten?  
     How can I boost my immune system naturally?  
     What supplements are good for better sleep?
"""


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the best health advice for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)