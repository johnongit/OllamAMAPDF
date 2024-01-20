from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sys import argv




llm = Ollama(model="mistral:latest")
llmInstruct = Ollama(model="mistral:instruct")
llmEmbed = OllamaEmbeddings(model="llama2:7b")

# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=llmEmbed)


# Fonction pour nettoyer l'input utilisateur
def nettoyer_input(input_str):
    return input_str.strip()


# Prompt
prompt = PromptTemplate.from_template(
    "<s>[INST] Utilisez les éléments de contexte suivants pour répondre à la question en Français à la fin. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, ne tentez pas d'inventer une réponse. Faites une réponse détaillée en trois phrases. Dites toujours \"merci d'avoir posé la question !\" à la fin de la réponse: {context} Réponds à la question : {query} [/INST]"
)

retriever = vectorstore.as_retriever(verbose=True, search_kwargs={"k": 4},search_type="similarity")




# create function name chatbot_loop
def chatbot_loop():
    print("Salut je suis un chatbot, pose moi une question !")
    #
    while True:
        input_str = input("> ")
        input_str = nettoyer_input(input_str)
        if input_str == "exit":
            break
        print(input_str)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            verbose=True,
            return_source_documents=True,
        )
        #chain.run(input_str)
        result = chain({"query": input_str})
        print(result["result"])


# if first arg is embed then embed the file placed in second arg
try:
    if argv[1] == "embed":
        if argv[2] == "":
            print("Please provide a file to embed.")
            exit()
        print("Embedding...")

        loader = PyPDFLoader(argv[2])
        pages = loader.load_and_split()
        db = Chroma.from_documents(pages, llmEmbed, persist_directory="./chroma_db")
        print("Done.")
        exit()
except:
    pass
chatbot_loop()