import requests
import feedparser
from langchain_community.document_loaders import NewsURLLoader
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS

# Function to fetch the latest news articles from BBC
def fetch_latest_bbc_news():
    url = 'http://feeds.bbci.co.uk/news/rss.xml'
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:5]:  # Fetch the latest 5 articles
        articles.append(entry.link)
    return articles

# Function to classify and summarize news articles
def classify_and_summarize_news(urls):
    loader = NewsURLLoader(urls)
    documents = loader.load()

    # Prompt template for the classification task
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for text classification tasks.
        Use the following pieces of retrieved context to classify the text. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Text: {text}
        Context: {context}
        Classification: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["text", "context"],
    )

    # Initializes a ChatGroq language model with the specified temperature, model name, and API key.
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key='***INSERT API KEY HERE***')

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    classification_chain = prompt | llm | StrOutputParser()

    summaries = []
    for doc in documents:
        classification = classification_chain.invoke({"text": doc.page_content, "context": format_docs(documents)})
        summary = classification['output'][:3]  # Summarize to 3 sentences
        summaries.append(summary)
    return summaries

# Function to convert text to speech
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en', tld='co.in')
    tts.save(filename)

if __name__ == '__main__':
    urls = fetch_latest_bbc_news()
    summaries = classify_and_summarize_news(urls)
    for i, summary in enumerate(summaries):
        text_to_speech(summary, f'static/summary_{i}.mp3')
