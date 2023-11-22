import gradio as gr
import os
import time

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter


from langchain.llms import OpenAI

from langchain.embeddings import OpenAIEmbeddings


from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain

def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, open_ai_key):
    if openai_key is not None:
        os.environ['OPENAI_API_KEY'] = open_ai_key
        loader = OnlinePDFLoader(pdf_doc.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        global qa 
        qa = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0.5), 
            retriever=retriever, 
            return_source_documents=False)
        return "Ready"
    else:
        return "You forgot OpenAI API key"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    history[-1][1] = ""
    
    for character in response:     
        history[-1][1] += character
        time.sleep(0.05)
        yield history
    

def infer(question, history):
    
    res = []
    for human, ai in history[:-1]:
        pair = (human, ai)
        res.append(pair)
    
    chat_history = res
    #print(chat_history)
    query = question
    result = qa({"question": query, "chat_history": chat_history})
    #print(result)
    return result["answer"]

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF • OpenAI</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;) <br />
    This version is set to store chat history, and uses OpenAI as LLM, don't forget to copy/paste your OpenAI API key</p>
</div>
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        
        with gr.Column():
            openai_key = gr.Textbox(label="You OpenAI API key", type="password")
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")
        
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send Message")
    load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    load_pdf.click(pdf_changes, inputs=[pdf_doc, openai_key], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot)

demo.launch()