import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient
import uuid
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from typing import List, Dict, TypedDict, Any, Optional
from pprint import pprint
from pinecone import Pinecone
import json
import logging
from dotenv import load_dotenv
import gc
import numpy as np
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN"),
)

class HuggingFaceInferenceEmbeddings:
    """Custom embedding class using HuggingFace Inference API instead of local model loading."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model_name = model_name
        self.client = client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using HuggingFace Inference API."""
        embeddings = []
        for text in texts:
            try:
                result = self.client.feature_extraction(text, model=self.model_name)
                if isinstance(result, np.ndarray):
                    embedding = result.tolist()
                elif isinstance(result, list):
                    # If it's already a list, check if it contains numpy arrays
                    if len(result) > 0 and isinstance(result[0], np.ndarray):
                        embedding = result[0].tolist()
                    elif len(result) > 0 and isinstance(result[0], list):
                        embedding = result[0]
                    else:
                        embedding = result
                else:
                    embedding = result
                
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding text: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 1024)  # multilingual-e5-large has 1024 dimensions
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using HuggingFace Inference API."""
        try:
            result = self.client.feature_extraction(text, model=self.model_name)
            
            # Convert numpy array to list if needed
            if isinstance(result, np.ndarray):
                return result.tolist()
            elif isinstance(result, list):
                # If it's already a list, check if it contains numpy arrays
                if len(result) > 0 and isinstance(result[0], np.ndarray):
                    return result[0].tolist()
                elif len(result) > 0 and isinstance(result[0], list):
                    return result[0]
                else:
                    return result
            else:
                return result
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1024  # multilingual-e5-large has 1024 dimensions

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
    source: List[str]


class RAGAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API")
        self.legal_index_name = "apna-waqeel3"
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.legal_index_name)
        self.llm = ChatGroq(
            temperature=0, model="gemma2-9b-it"
        )
        self.vectorstore = None
        self.retriever = None
        self.web_search_tool = TavilySearchResults(k=3)
        self._initialize_vectorstore()
        self._initialize_prompts()
        gc.collect()
        self.max_context_length = 4096
        self.memory = ConversationBufferWindowMemory(k=5)
        self.chats_loaded = False

    def _initialize_vectorstore(self):
        embeddings = HuggingFaceInferenceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.vectorstore = PineconeVectorStore(index=self.index, embedding=embeddings)
        self.retriever = self.vectorstore.as_retriever()
        print("Vectorstore initialized using HuggingFace Inference API.")

    def _initialize_prompts(self):
        self.analyze_sentiment_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a legal assistant analyzing the sentiment of text. First, determine if the input text is related to a legal case or legal matter.
    If the text is NOT related to legal matters, or the text is general query that does not need to be adressed by any lawyer then return exactly: None

    If the text IS legal-related, analyze its sentiment and categorize it into exactly one of these categories: {categories}

    Text: {text}

    Return only 'None' for non-legal text, or just the category name for legal text without any additional explanation. <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["categories", "text"],
    )
        self.analyze_sentiment = self.analyze_sentiment_prompt | self.llm | StrOutputParser()        
        self.retrieval_grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
             <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "document"],
        )
        
        self.retrieval_grader = (
            self.retrieval_grader_prompt | self.llm | JsonOutputParser()
        )
        self.generate_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a legal assistant for question-answering tasks in the context of Pakistani law. Structure your responses in Markdown format to aid in legal research. Bold the key terms like section or laws and use bullet points for lists. Provide a detailed response to the user's question.

        Previous conversation context:
        {chat_history}

        Question: {question} 
        Retrieved information: {context} 

        Remember to:
        1. Use Markdown headers (# for main title, ## for sections)
        2. Use ordered list for steps
        3. Use bold (**) for emphasis on important terms
        4. Include relevant legal citations where applicable
        5. Use proper line breaks and formatting

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "context", "chat_history"],
        )
        self.rag_chain = self.generate_prompt | self.llm | StrOutputParser()

        self.hallucination_grader_prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )
        self.hallucination_grader = (
            self.hallucination_grader_prompt | self.llm | JsonOutputParser()
        )

        self.answer_grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
             <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )
        self.answer_grader = self.answer_grader_prompt | self.llm | JsonOutputParser()

        self.question_router_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert legal assistant specializing in routing user questions.
For questions about case law, legal statutes, contracts, or legal research topics, use vectorstore.
For general inquiries or factual questions, use web search.
Return only one of these two exact strings: "vectorstore" or "web_search"

Question to route: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question"],
        )

        self.question_router = (
            self.question_router_prompt | self.llm | StrOutputParser()
        )

    def generate(self, state: Dict) -> Dict:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        sources = []
        chat_context = self.memory.load_memory_variables({})["history"]

        doc_texts = []
        total_length = len(question) + len(chat_context)
        for doc in documents:
            doc_text = doc.page_content[:1000]
            if total_length + len(doc_text) > self.max_context_length:
                break
            doc_texts.append(doc_text)
            total_length += len(doc_text)
            # Extract source from document metadata if available
            if isinstance(doc, Document):
                if "source" in doc.metadata:
                    if not doc.metadata["source"].startswith("/kaggle"):                        
                        sources.append(doc.metadata["source"])
                elif "file_name" in doc.metadata:
                    sources.append(doc.metadata["file_name"])

        context = "\n".join(doc_texts)
        gc.collect()
        try:
            generation = self.rag_chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_context
            })
            if any(phrase in generation.lower() for phrase in [
                "i apologize", "i'm sorry", "i do not have the capability",
                "cannot assist", "cannot help", "not able to"
            ]):
                print("---DETECTED ERROR MESSAGE, TRYING WEB SEARCH---")
                web_results = self.web_search(state)
                if web_results["documents"]:
                    state["documents"] = web_results["documents"]
                    return self.generate(state)  # Retry with web results

            final_answer = f"{generation}"
            del generation
            gc.collect()
            return {
                "documents": documents,
                "question": question,
                "generation": final_answer,
                "source": sources
            }
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return {
                "documents": documents,
                "question": question,
                "generation": "I apologize, but I encountered an error. Please try rephrasing your question.",
                "source": []
            }

    def retrieve(self, state: Dict) -> Dict:
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        documents = documents[:5] if len(documents) > 5 else documents
        gc.collect()
        return {
            "documents": documents,
            "question": question,
            "source": [],  # Initialize empty source list
        }

    def _safe_parse_json(self, text: str) -> dict:
        """Enhanced JSON parsing with better error handling"""
        try:
            if isinstance(text, dict):
                if "score" in text:
                    return {"score": str(text["score"]).lower()}
                return text
            if "<tool_call>" in text:
                # Extract JSON between first { and last }
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = text[start:end]
                    try:
                        data = json.loads(json_str)
                        # Handle tool call specific structure
                        if "arguments" in data:
                            args = data["arguments"]
                            # Look for different types of relevance indicators
                            if "score" in args:
                                return {"score": str(args["score"]).lower()}
                            if "document" in args and "userQuestion" in args:
                                # Assume relevance if properly formatted
                                return {"score": "yes"}
                        return {"score": "yes"}  # Default to yes if well-formed
                    except json.JSONDecodeError:
                        pass

            # Try parsing as regular JSON
            if text.startswith("{") and text.endswith("}"):
                data = json.loads(text)
                if "score" in data:
                    return {"score": str(data["score"]).lower()}

            # Handle plain text responses
            text_lower = text.lower().strip()
            if any(word in text_lower for word in ["yes", "relevant", "true"]):
                return {"score": "yes"}
            
            return {"score": "no"}

        except Exception as e:
            logging.error(f"JSON parsing error: {e}")
            return {"score": "no"}

    def grade_documents(self, state: Dict) -> Dict:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        source = state["source"]
        filtered_docs = []
        web_search = "No"

        for d in documents:
            try:
                # Clean document content
                d.page_content = d.page_content.replace("\\", "").replace('"', "")
                print(f"Document: {d.page_content[:100]}...")
                # Get grading response
                response = self.retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                
                # Handle response parsing
                if isinstance(response, dict):
                    grade = response.get("score", "no")
                else:
                    parsed = self._safe_parse_json(response)
                    grade = parsed.get("score", "no")

                grade = str(grade).lower()
                
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search = "Yes"

            except Exception as e:
                logging.error(f"Document grading error: {e}")
                web_search = "Yes"
                continue

        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
            "source": source,
        }

    def web_search(self, state: Dict) -> Dict:
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])
        sources = state.get("source", [])  # Get existing sources or initialize empty list

        try:
            docs = self.web_search_tool.invoke({"query": question})

            # Limit results to prevent memory issues
            if isinstance(docs, list):
                docs = docs[:3]

            # Convert search results to Document objects
            web_documents = []
            for doc in docs:
                web_documents.append(
                    Document(
                        page_content=f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}",
                        metadata={
                            "source": doc.get("url", ""),
                            "score": doc.get("score", 0),
                        },
                    )
                )
            if documents:
                web_documents.extend(documents)

        except Exception as e:
            print(f"Web search error: {e}")
            web_documents = documents
        gc.collect()
        return {
            "documents": web_documents,
            "question": question,
            "source": sources,  # Maintain source state
        }

    def route_question(self, state: Dict) -> str:
        print("---ROUTE QUESTION---")
        question = state["question"]
        try:
            route = "vectorstore"
            if route == "web_search":
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "websearch"
            else:
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"
        except Exception as e:
            print(f"Routing error: {e}, defaulting to vectorstore")
            return "vectorstore"

    def decide_to_generate(self, state: Dict) -> str:
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        if (web_search == "Yes"):
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: Dict) -> str:
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        attempts = state.get("attempts", 0)
        
        if attempts >= 2:
            print("---MAX RETRIES REACHED, RETURNING CURRENT GENERATION---")
            return "useful"

        try:
            # Check for error messages in generation
            if any(phrase in generation.lower() for phrase in [
                "i apologize", "i'm sorry", "i do not have the capability",
                "cannot assist", "cannot help", "not able to"
            ]):
                print("---DETECTED ERROR MESSAGE IN GENERATION---")
                return "not useful"

            score = self.hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            
            if isinstance(score, str):
                parsed_score = self._safe_parse_json(score)
            else:
                parsed_score = score

            grade = parsed_score.get("score", "no").lower()
            
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                try:
                    score = self.answer_grader.invoke(
                        {"question": question, "generation": generation}
                    )
                    
                    if isinstance(score, str):
                        parsed_score = self._safe_parse_json(score)
                    else:
                        parsed_score = score
                        
                    grade = parsed_score.get("score", "no").lower()
                except Exception as e:
                    logging.error(f"Answer grading error: {e}")
                    grade = "no"
                
                if grade == "yes":
                    return "useful"
                return "not useful" if attempts < 2 else "useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                attempts += 1
                return "not supported" if attempts < 2 else "useful"
                
        except Exception as e:
            logging.error(f"Generation grading error: {e}")
            # If we get an error and the generation looks like an error message,
            # try again rather than accepting it
            if "sorry" in generation.lower() or "apologize" in generation.lower():
                return "not supported" if attempts < 2 else "useful"
            return "useful"

    def build_workflow(self):
        workflow = StateGraph(state_schema=GraphState)

        # Add nodes with attempt tracking capabilities
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node(
            "generate",
            lambda x: {**x, "attempts": x.get("attempts", 0) + 1} | self.generate(x),
        )

        # Set entry point
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        # Add basic edges
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",  # Direct to generate instead of update_query
            },
        )

        workflow.add_edge("websearch", "generate")

        # Add conditional edges for generation grading
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {"not supported": "generate", "not useful": "websearch", "useful": END},
        )

        return workflow.compile()

    def run(
        self,
        question: str,
        chat_history: List[Dict] = None,
    ) -> Dict[str, Any]:
        try:
            if chat_history is None:
                chat_history = []
            else:
                for message in chat_history:
                    # Handle pyodbc.Row objects by accessing columns directly
                    try:
                        # Try dictionary access first
                        role = message.get('Type') if isinstance(message, dict) else message.role
                        content = message.get('content') if isinstance(message, dict) else message.content
                        
                        if role == 'Human Message' or role == 'user':
                            self.memory.save_context({'input': content}, {'output': ''})
                        elif role == 'AI Message' or role == 'assistant':
                            self.memory.save_context({'input': ''}, {'output': content})
                    except AttributeError:
                        # If message is a tuple/row, try positional access
                        # Assuming the order is (role, content, ...)
                        if isinstance(message, tuple):
                            role, content = message[0], message[1]
                            if role in ('Human Message', 'user'):
                                self.memory.save_context({'input': content}, {'output': ''})
                            elif role in ('AI Message', 'assistant'):
                                self.memory.save_context({'input': ''}, {'output': content})
            VALID_CATEGORIES = {
            "Civil", "Criminal", "Corporate", "Constitutional", "Tax",
            "Family", "Intellectual Property", "Labor and Employment",
            "Immigration", "Commercial", "Environmental", "Banking and Finance",
            "Cyber Law", "Alternate Dispute Resolution"
        }
            chat_context = self.memory.load_memory_variables({})["history"]
            sentiment = self.analyze_sentiment.invoke(
                {
                 "text": question,
                 "categories": ", ".join(sorted(VALID_CATEGORIES))
                }
            )
            print(f"Sentiment: {sentiment}")
            app = self.build_workflow()
            inputs = {
                "question": question,
                "documents": [],
                "source": [],
            }
            last_output = None
            try:
                for output in app.stream(inputs):
                    for key, value,  in output.items():
                        pprint(f"Finished running: {key}:")
                        print("Output:")
                        last_output = value                        
                    gc.collect()
                result = last_output["generation"]
                logging.info(f"Result: {result}")
                response_text = result if isinstance(result, str) else str(result)
                response = {
                    "chat_response": response_text,
                    "references": last_output["source"],
                    "Sentiment": sentiment,
                }
                return response

            except Exception as e:
                print(f"Error in RAG workflow: {e}")
                return {
                    "chat_response": "I apologize, but I encountered an error processing your request.",
                    "references": [],
                    "recommended_lawyers": [],
                }

        finally:
            gc.collect()

    def retrieve_dataset(self):
        """Retrieve all documents from the Pinecone index"""
        try:
            # Use vectorstore's similarity search to get all documents
            documents = self.vectorstore.similarity_search(query="", k=100)
            documents = [
                doc
                for doc in documents
                if doc.page_content and doc.page_content.strip()
            ]
            print(f"Retrieved {len(documents)} documents from vectorstore")
            return documents
        except Exception as e:
            print(f"Error retrieving dataset: {e}")
            return []

    def save_context(self, user_input: str, assistant_output: str):
        self.memory.save_context({"input": user_input}, {"output": assistant_output})


if __name__ == "__main__":
    rag = RAGAgent()
    query = "What is the procedure for filing a lawsuit in Pakistan?"
    result = rag.run(query, [])
    print(result)
