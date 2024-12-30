from langchain_community.document_loaders import (
    AssemblyAIAudioTranscriptLoader,
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube, extract
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import os

load_dotenv()



api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
        
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
)

class DocumentLoaderError(Exception):
    """Custom exception for document loading errors"""
    pass

class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass

class BaseDocumentLoader:
    """Base class for document loaders"""
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(self.__class__.__name__)
    
    def load(self):
        """Abstract method to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement load() method")

class AudioLoader(BaseDocumentLoader):
    """Class for handling audio file transcription"""
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.getenv("ASSEMBLY_API_KEY")
        if not self.api_key:
            raise DocumentLoaderError("Assembly AI API key not found")

    def load(self, file_path):
        try:
            loader = AssemblyAIAudioTranscriptLoader(
                file_path=file_path,
                api_key=self.api_key
            )
            docs = loader.load()
            return {
                'text': docs[0].page_content,
                'metadata': docs[0].metadata
            }
        except Exception as e:
            raise DocumentLoaderError(f"Error transcribing audio: {str(e)}")

class TextDocumentLoader(BaseDocumentLoader):
    """Class for handling text and PDF documents"""
    def load(self, file_path):
        try:
            if file_path.endswith('.pdf'):
                return self._load_pdf(file_path)
            else:
                return self._load_text(file_path)
        except Exception as e:
            raise DocumentLoaderError(f"Error loading document: {str(e)}")

    def _load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return {'pages': pages, 'page_count': len(pages)}

    def _load_text(self, file_path):
        loader = TextLoader(file_path)
        data = loader.load()
        return {'content': data}

class WebpageLoader(BaseDocumentLoader):
    """Class for handling webpage content"""
    def load(self, url):
        if not url.startswith(('http://', 'https://')):
            raise DocumentLoaderError("Invalid URL format")
        
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            content = docs[0].page_content.strip()
            
            with open("output.txt", "w", encoding='utf-8') as file:
                file.write(content)
                file.write("\n\n")
            
            return {'content': content}
        except Exception as e:
            raise DocumentLoaderError(f"Error loading webpage: {str(e)}")

class YoutubeTranscriptLoader(BaseDocumentLoader):
    """Class for handling YouTube video transcripts"""
    def __init__(self):
        super().__init__()
        load_dotenv()


    def load(self, video_url):
        video_id = self._get_video_id(video_url)
        if not video_id:
            raise DocumentLoaderError("Invalid YouTube URL")
        
        transcript = self._fetch_transcript(video_id)
        if not transcript:
            raise TranscriptionError("No transcript available")
        
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        return {'transcript': transcript_text}

    def _get_video_id(self, url):
        try:
            return extract.video_id(url)
        except Exception as e:
            self.logger.error(f"Error extracting video ID: {e}")
            return None

    def _fetch_transcript(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            try:
                transcript = transcript_list.find_transcript(['en'])
                return transcript.fetch()
            except NoTranscriptFound:
                return self._handle_non_english_transcript(transcript_list)
                
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            self.logger.error(f"Transcript error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return None

    def _handle_non_english_transcript(self, transcript_list):
        try:
            available_transcript = transcript_list.find_generated_transcript(
                ['en', 'ja', 'ko', 'es', 'fr', 'de', 'hi']
            )
            if available_transcript:
                self.logger.info("Found non-English transcript. Translating...")
                return self._translate_transcript(available_transcript.fetch())
        except Exception as e:
            self.logger.error(f"Error handling non-English transcript: {str(e)}")
        return None

    def _translate_transcript(self, transcript):
        try:
            full_text = ' '.join([entry['text'] for entry in transcript])
            prompt = f"Translate the following transcript into English:\n{full_text}"
            
            response = gemini.generate_content(prompt)
            translated_text = response.text.strip()
            
            return [{"text": line} for line in translated_text.split('\n') if line.strip()]
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return None

def main():
    # Example usage
    try:
        transcripts = ""
        # Audio loading example
        audio_loader = AudioLoader()
        audio_result = audio_loader.load("./What is Retrieval-Augmented Generation (RAG)？.mp3")
        print("Audio transcript:", audio_result['text'])
        transcripts += audio_result['text']
        print("Audio transcript Success")

        # Document loading example
        doc_loader = TextDocumentLoader()
        pdf_result = doc_loader.load("./LangChain_Cheat_Sheet_KDnuggets.pdf")
        print("PDF pages:", pdf_result['pages'], "length: ", len(pdf_result['pages']))
        transcripts += str(pdf_result['pages'])
        print("PDF transcript Success")

        # Webpage loading example
        web_loader = WebpageLoader()
        web_result = web_loader.load("https://medium.com/pythoneers/building-a-multi-agent-system-using-crewai-a7305450253e")
        print("Webpage content length:", len(web_result['content']))
        transcripts += web_result['content']
        print("Webpage transcript Success")

        # YouTube transcript loading example
        youtube_loader = YoutubeTranscriptLoader()
        video_url = "https://youtu.be/T-D1OfcDW1M?si=WOlSKx3YXpWQgJ1m"
                
        transcript_result = youtube_loader.load(video_url)
        print("Transcript:", transcript_result['transcript'])
        transcripts += transcript_result['transcript']
        print("Youtube Transcript Success")

        print(f"Gemini Tokens: {gemini.count_tokens(transcripts)}.")

    except (DocumentLoaderError, TranscriptionError) as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()