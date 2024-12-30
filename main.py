from loaders import AudioLoader, YoutubeTranscriptLoader, TextDocumentLoader, WebpageLoader
from loaders import gemini

# ytube_video_url = input('URL: ')
# text_document_url = input('Text Document URL: ')
# webpage_url = input('Webpage URL: ')
# audio_url = input('Audio URL: ')


ytube_video_url = "https://youtu.be/T-D1OfcDW1M?si=ROLomh36SaVhY9mo"
text_document_url = "./LangChain_Cheat_Sheet_KDnuggets.pdf"
webpage_url = "https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c"
audio_url = "./What is Retrieval-Augmented Generation (RAG)？.mp3"

# Initialize Loaders
youtube_loader = YoutubeTranscriptLoader()
audio_loader = AudioLoader()
text_document_loader = TextDocumentLoader()
webpage_loader = WebpageLoader()

# Load transcript from YouTube
if ytube_video_url is not None:
    transcript = youtube_loader.load(ytube_video_url)

# Load Webpage
if webpage_url is not None:
    web_content = webpage_loader.load(webpage_url)

# Load Text Document
if text_document_url is not None:
    text_content = text_document_loader.load(text_document_url)

# Audio Loader
if audio_url is not None:
    audio_content = audio_loader.load(audio_url)

context = str(transcript) + '\n' + str(web_content) + '\n' + str(text_content) + '\n' + str(audio_content)

while True:
    query = input('Query: ')
    if query == 'exit':
        break

    prompt = f"""Answer the <>{query}</> based on the context, context: {context}"""
    response = gemini.generate_content(query)
    print(response.text)
    print('Tokens:', gemini.count_tokens(response.text))
