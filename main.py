import os
import gradio as gr


from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Terminal Colors
RED = "\033[31;1m"
GREEN = "\033[32;1m"
BLUE = "\033[34;1m"
PUPRLE = "\033[35;1m"
WHITE = "\033[37;1m"


load_dotenv()

class ConversationalRAGChatbot:
    def __init__(self):
        self.setup_llm()
        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_chains()
        self.store ={}
    
    def setup_llm(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192"
        )   
        
    def setup_embeddings(self):
        hf_token = os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN_ZETA")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            
        self.embeddings = HuggingFaceEmbeddings(
            model="all-MiniLM-L6-v2",
            model_kwargs = {"device":"cpu"},
            encode_kwargs={"normalize_embeddings":True}
            )
        
    def setup_vector_store(self):
        try:
            # Check if the file exists
            pdf_path = os.path.join("Indus Valley Annual Report 2025.pdf")
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            print(f"{BLUE}Loading PDF document...")

            # Load the document and create Vector Store
            loader = PyPDFLoader(pdf_path)
            docs  = loader.load()

            if not docs:
                raise ValueError("No documents loaded from the PDF")

            print(f"{GREEN}Loaded {len(docs)} pages from PDF.")

            # Splitting Documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            print(f"{BLUE}Splitting the documents into chunks...")
            splits = text_splitter.split_documents(docs)
            print(f"{GREEN}Created {len(splits)} document chunks")

            # Clean and validate document chunks 
            print(f"{BLUE}Cleaning document chunks...")

            cleaned_splits = []

            for idx, doc in enumerate(splits):
                try:
                    if hasattr(doc, "page_content") and doc.page_content:
                        # Convert to string and clean 
                        content = str(doc.page_content).strip() 

                        # Remove null characters
                        content = content.replace("\x00", "").replace("\ufffd", "")

                        # Skip if content is too short
                        if len(content) < 10:
                            continue

                        content = content.encode("utf-8", errors="ignore").decode("utf-8")

                        # Update the document content
                        doc.page_content = content
                        cleaned_splits.append(doc)
                except Exception as e:
                    print(f"{RED}Skipping problematic chunk {idx}: {str(e)}")

            if not cleaned_splits:
                raise ValueError("No valid document chunks created after cleaning.")
            
            print(f"{GREEN}Cleaned chunks: {len(cleaned_splits)} valid chunks.")

            # Create vector store in batches to handle large documents
            print(f"{BLUE}Creating vector store...")
            batch_size = 50 
            all_texts = []
            all_metadatas = []

            for doc in cleaned_splits:
                all_texts.append(doc.page_content)
                all_metadatas.append(doc.metadata if hasattr(doc, "metadata") else {})

            # Create vector store using from_texts method for better control
            self.vector_store = Chroma.from_texts(
                texts=all_texts,
                embedding=self.embeddings,
                metadatas=all_metadatas,
                persist_directory="./chroma_db"
            )

            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs = {"k":4}
            )

            print(f"{GREEN}Vector store created successfully.")

        except Exception as e:
            print(f"{RED}Error in stepup_vector_store: {str(e)}")
            raise e
        
    def setup_chains(self):
        
        contextualize_q_system_prompt = (
            "give a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question"
            "just reformulate it if needed and otherwise return it as is."
        )   
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]) 
        
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,  
            self.retriever,
            contextualize_q_prompt
        )
        
        qa_system_prompt = (
            "You are an assistant for question-answering tasks about perspective on the Indian startup ecosystem. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
            
        ])
        
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            prompt=qa_prompt
        )
        rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            question_answer_chain
        )
        
        self.conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    def get_session_history(self, session_id:str) -> BaseChatMessageHistory:
        """Get or create session history"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id] 
    
    def chat(self, message, history, session_id="default_session"):
        try:
            response = self.conversation_rag_chain.invoke(
                {"input": message},
                config = {"configurable": {"session_id":session_id}}
            )   
            
            return response["answer"]
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    def clear_history(self, session_id="default_session"):
        if session_id in self.store:
            self.store[session_id] = ChatMessageHistory()
        return "chat history cleared!"        
    
print(f"{GREEN}Initializing chatbot...")
chatbot = ConversationalRAGChatbot()
print(f"{GREEN} chatbot initialized successfully")

def chat_interface(message, history):
    response = chatbot.chat(message, history=history)
    return response

def clear_chat():
    return chatbot.clear_history()

with gr.Blocks(
    title="Indus Valley Report 2025 PDF Q&A chatbot",
    theme=gr.themes.Soft()
    
) as demo:
    gr.Markdown(
        """
        # 🤖 GROQ Indus Valley Report 2025 PDF Q&A Chatbot
        Ask me anything about Indus Valley Report 2025! This chatbot has knowledge from
        comprehensive source about growth in Indian Startups and can maintain the conver-
        sation context.
        
        ** Topics I can help with:**
        -India-The Last Five Years
        -Long-Term Structural Forces
        -Consumption
        -Funding Trends
        -IPO Boom
        -Sector Deep Dives
        -Play Book
        """
    )
    
    chatbot_interface = gr.ChatInterface(
            fn=chat_interface,
            title="Chat with Startup Expert",
            description="Ask question related to Economy and Startups",
            examples=[
                "Discuss how the Indian economy transitioned from the COVID-induced downturn to a consumption-led recovery. What macroeconomic levers were used during this period?",
                "Analyze the structural reasons why India's manufacturing sector remains underwhelming despite recent government incentives like PLIs and import bans.",
                "How does India1 influence the country's consumer market and equity trends? Explore the implications of a 'deepening not widening' India1 class.",
                "Explain why gold plays such a dominant role in India's household wealth and loan collateral structure, especially in rural economies.",
                "Evaluate how Digital Public Infrastructure (DPI) has transformed India into a digital welfare state. Which DPI initiatives have had the most impact?"
            ],
            cache_examples=False,
        )

    with gr.Row():
            gr.Markdown(
                """
                ### 💡Tips:
                - Ask follow-up questions - I remember our conversation!
                - be specific about what aspect you're interested in
                - You can ask for comparisions between different approaches
                - Try asking "Tell me mire about that" after any response
                """
            )
            

# Launch the Interface 
if __name__ == "__main__":
    print(f"{WHITE}Starting Gradio interface...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )        