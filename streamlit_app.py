import streamlit as st
import torch
import torch.nn as nn
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import string

# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================
# Streamlit page config must be the first command[cite: 9]
st.set_page_config(page_title="Support Bot", layout="wide") #[cite: 9]
st.title("🤖 Multi-Agent Support Bot") #[cite: 9]

# ==========================================
# 2. PYTORCH CLASSIFIER SETUP
# ==========================================
@st.cache_resource # Cache the model so it doesn't reload on every chat message
def load_classifier():
    checkpoint = torch.load(r'D:\classifier.pt', weights_only=False)#[cite: 5]
    vocab = checkpoint['vocab']
    label_map = checkpoint['label_map']

    class IntentClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=4, n_layers=1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x, lengths):
            embedded = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.rnn(packed)
            h = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.fc(self.dropout(h))

    model = IntentClassifier(vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, vocab, label_map

model, vocab, label_map = load_classifier()

def classify_intent(text):
    clean_text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    indices = [vocab.get(token, vocab["<unk>"]) for token in clean_text][:40]
    if not indices: indices = [vocab["<unk>"]]
    
    seq = torch.tensor([indices])
    length = torch.tensor([len(indices)])
    
    with torch.no_grad():
        output = model(seq, length)
        probs = torch.softmax(output, dim=1)[0]
        confidence, label_idx = torch.max(probs, dim=0)
        
    return label_map[label_idx.item()], confidence.item() #[cite: 7]

# ==========================================
# 3. KNOWLEDGE BASE & TOOLS
# ==========================================
@st.cache_resource
def load_retriever():
    raw_docs = [
        Document(page_content="Returns are accepted within 30 days. Laptops must be unopened."),
        Document(page_content="Refunds take 5-7 business days to reflect on your original card."),
        Document(page_content="Warranty covers manufacturing defects for 1 year, not accidental damage.")
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #[cite: 7]
    vectorstore = FAISS.from_documents(raw_docs, embeddings) #[cite: 7]
    return vectorstore.as_retriever(search_kwargs={"k": 2}) #[cite: 7]

retriever = load_retriever()

@tool
def search_docs(query: str) -> str:
    """Search the company knowledge base for policy and product info.""" #[cite: 6]
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def check_invoice(customer_id: str) -> str:
    """Look up a customer's latest billing statement and balance.""" #[cite: 6]
    return f"Customer {customer_id} owes $0.00. Last payment cleared."

@tool
def track_order(order_id: str) -> str:
    """Track the physical location and delivery date of an order."""
    return f"Order {order_id} is in transit. Arriving tomorrow by 8 PM."

# ==========================================
# 4. LANGCHAIN ORCHESTRATION
# ==========================================
try:
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.3, 
        api_key="GROQ_API_KEY"
    )
except:
    llm = ChatOllama(model="llama3.2:3b", temperature=0.3)

def build_agent(system_prompt, agent_tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"), #[cite: 8]
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), #[cite: 6]
    ])
    agent = create_tool_calling_agent(llm, agent_tools, prompt)
    return AgentExecutor(agent=agent, tools=agent_tools, verbose=False)

AGENTS = {
    "billing": build_agent("You are the billing agent. Use check_invoice to confirm balances before answering.", [check_invoice]),
    "order_status": build_agent("You are the shipping agent. Use track_order to get real-time location data.", [track_order]),
    "returns_refunds": build_agent("You are the returns agent. Use search_docs to verify return policies.", [search_docs]),
    "general": build_agent("You are a general agent. Use search_docs if helpful; otherwise answer directly.", [search_docs]) #[cite: 7]
}

def route_and_run(user_message: str, history: list):
    intent_label, confidence = classify_intent(user_message) #[cite: 7]
    
    if confidence < 0.60:
        intent_label = "general" #[cite: 7]
        
    active_agent = AGENTS.get(intent_label, AGENTS["general"])
    
    # Format memory for LangChain[cite: 8]
    formatted_history = [("user" if msg["role"] == "user" else "assistant", msg["content"]) for msg in history]
    
    result = active_agent.invoke({
        "input": user_message, 
        "chat_history": formatted_history 
    })
    
    return {
        "reply": result["output"],
        "intent": intent_label,
        "confidence": confidence,
        "agent": intent_label
    }

# ==========================================
# 5. STREAMLIT UI LAYOUT
# ==========================================
# Setup the sidebar for live diagnostic data[cite: 9]
with st.sidebar:
    st.subheader("Routing") #[cite: 9]
    intent_placeholder = st.empty() #[cite: 9]
    conf_placeholder = st.empty() #[cite: 9]
    agent_placeholder = st.empty() #[cite: 9]

# Initialize standard chat interface memory[cite: 9]
if "history" not in st.session_state:
    st.session_state.history = [] #[cite: 9]

# Render full conversation history[cite: 9]
for turn in st.session_state.history:
    st.chat_message(turn["role"]).write(turn["content"]) #[cite: 9]

# Handle incoming user message[cite: 9]
if user_msg := st.chat_input("How can I help?"): #[cite: 9]
    st.chat_message("user").write(user_msg) #[cite: 9]
    
    # Process the message through your routing logic
    result = route_and_run(user_msg, st.session_state.history) #[cite: 9]
    
    # Display the LLM reply[cite: 9]
    st.chat_message("assistant").write(result["reply"]) #[cite: 9]
    
    # Update sidebar with live classifier output and agent assignment[cite: 9]
    intent_placeholder.metric("Predicted intent", result["intent"]) #[cite: 9]
    conf_placeholder.progress(result["confidence"]) #[cite: 9]
    agent_placeholder.write(f"Handled by: {result['agent']} agent") #[cite: 9]
    
    # Save to session state[cite: 9]
    st.session_state.history += [
        {"role": "user", "content": user_msg}, #[cite: 9]
        {"role": "assistant", "content": result["reply"]}, #[cite: 9]
    ]