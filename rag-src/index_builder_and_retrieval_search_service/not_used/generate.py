### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import logging
                           
import rag_response_service.document_retrieval_grader
from utils.string_util import str_limit

logger = logging.getLogger(__name__)

def generate_answer(messages, documents):
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    logger.debug(f"prompt: {prompt}")
    # prompt: input_variables=['context', 'question'] metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))]

    # LLM
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # cheaper
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # cheaper

    # get context (docs)
    question = messages[-1].content
    relevant_docs = answer_service.document_retrieval_grader.get_relevant_documents(question)

    # Post-processing
    def format_docs(docs):
        for doc in docs:
            logger.debug(f"docs_context: {str_limit(doc.page_content)}")
        return "\n\n".join(doc.page_content for doc in docs)

    docs_context = format_docs(relevant_docs)

    logger.info(f"docs_context: {str_limit(docs_context)}")

    # Chain
    #rag_chain = prompt | llm  # | StrOutputParser()
    rag_chain = llm

    # Run
    generation = rag_chain.invoke({
        #"context": docs_context,
        #"question": question, 
        "messages": messages})
    logger.info(f"generation: {generation}")
    logger.info("generation done, length:"+str(len(generation)))
    #print(generation)

    return generation
