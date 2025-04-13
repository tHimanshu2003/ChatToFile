from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv
import gradio as gr
import markdowm as md
import base64

# Load environment variables
load_dotenv()

llm_models = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "tiiuae/falcon-7b-instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # "deepseek-ai/deepseek-vl2",  ## 54GB > 10GB
    # "deepseek-ai/deepseek-vl2-small",  ## 32GB > 10GB
    # "deepseek-ai/deepseek-vl2-tiny",  ## high response time
    # "deepseek-ai/deepseek-llm-7b-chat",  ## 13GB > 10GB
    # "deepseek-ai/deepseek-math-7b-instruct",  ## 13GB > 10GB
    # "deepseek-ai/deepseek-coder-33b-instruct",  ## 66GB > 10GB
    # "deepseek-ai/DeepSeek-R1-Zero",  ## 688GB > 10GB
    # "mistralai/Mixtral-8x22B-Instruct-v0.1",    ## 281GB>10GB
    # "NousResearch/Yarn-Mistral-7b-64k",  ## 14GB>10GB
    # "impira/layoutlm-document-qa",  ## ERR
    # "Qwen/Qwen1.5-7B",      ## 15GB
    # "Qwen/Qwen2.5-3B",      ## high response time
    # "google/gemma-2-2b-jpn-it",   ## high response time
    # "impira/layoutlm-invoices",   ## bad req
    # "google/pix2struct-docvqa-large",  ## bad req
    # "google/gemma-7b-it", ## 17GB > 10GB
    # "google/gemma-2b-it",  ## high response time
    # "HuggingFaceH4/zephyr-7b-beta",   ## high response time
    # "HuggingFaceH4/zephyr-7b-gemma-v0.1",     ## bad req
    # "microsoft/phi-2",    ## high response time
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     ## high response time
    # "mosaicml/mpt-7b-instruct",     ## 13GB>10GB
    # "google/flan-t5-xxl" ## high respons time
    # "NousResearch/Yarn-Mistral-7b-128k",  ## 14GB>10GB
    # "Qwen/Qwen2.5-7B-Instruct",     ## 15GB>10GB
]

embed_models = [
    "BAAI/bge-small-en-v1.5",  # 33.4M
    "NeuML/pubmedbert-base-embeddings",
    "BAAI/llm-embedder", # 109M
    "BAAI/bge-large-en" # 335M
]

# Global variable for selected model
selected_llm_model_name = llm_models[0]  # Default to the first model in the list
selected_embed_model_name = embed_models[0] # Default to the first model in the list
vector_index = None

# Initialize the parser
parser = LlamaParse(api_key=os.getenv("LLAMA_INDEX_API"), result_type='markdown')
# Define file extractor with various common extensions
file_extractor = {
    '.pdf': parser,  # PDF documents
    '.docx': parser,  # Microsoft Word documents
    '.doc': parser,  # Older Microsoft Word documents
    '.txt': parser,  # Plain text files
    '.csv': parser,  # Comma-separated values files
    '.xlsx': parser,  # Microsoft Excel files (requires additional processing for tables)
    '.pptx': parser,  # Microsoft PowerPoint files (for slides)
    '.html': parser,  # HTML files (web pages)
    # '.rtf': parser,  # Rich Text Format files
    # '.odt': parser,  # OpenDocument Text files
    # '.epub': parser,  # ePub files (e-books)

    # Image files for OCR processing
    '.jpg': parser,  # JPEG images
    '.jpeg': parser,  # JPEG images
    '.png': parser,  # PNG images
    # '.bmp': parser,  # Bitmap images
    # '.tiff': parser,  # TIFF images
    # '.tif': parser,  # TIFF images (alternative extension)
    # '.gif': parser,  # GIF images (can contain text)

    # Scanned documents in image formats
    '.webp': parser,  # WebP images
    '.svg': parser,  # SVG files (vector format, may contain embedded text)
}


# File processing function
def load_files(file_path: str, embed_model_name: str):
    try:
        global vector_index
        document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        vector_index = VectorStoreIndex.from_documents(document, embed_model=embed_model)
        print(f"Parsing done for {file_path}")
        filename = os.path.basename(file_path)
        return f"Ready to give response on {filename}"
    except Exception as e:
        return f"An error occurred: {e}"


# Function to handle the selected model from dropdown
def set_llm_model(selected_model):
    global selected_llm_model_name
    selected_llm_model_name = selected_model  # Update the global variable
    # print(f"Model selected: {selected_model_name}")
    # return f"Model set to: {selected_model_name}"


# Respond function that uses the globally set selected model
def respond(message, history):
    try:
        # Initialize the LLM with the selected model
        llm = HuggingFaceInferenceAPI(
            model_name=selected_llm_model_name,
            contextWindow=8192,  # Context window size (typically max length of the model)
            maxTokens=1024,  # Tokens per response generation (512-1024 works well for detailed answers)
            temperature=0.3,  # Lower temperature for more focused answers (0.2-0.4 for factual info)
            topP=0.9,  # Top-p sampling to control diversity while retaining quality
            frequencyPenalty=0.5,  # Slight penalty to avoid repetition
            presencePenalty=0.5,  # Encourages exploration without digressing too much
            token=os.getenv("TOKEN")
        )

        # Set up the query engine with the selected LLM
        query_engine = vector_index.as_query_engine(llm=llm)
        bot_message = query_engine.query(message)

        print(f"\n{datetime.now()}:{selected_llm_model_name}:: {message} --> {str(bot_message)}\n")
        return f"{selected_llm_model_name}:\n{str(bot_message)}"
    except Exception as e:
        if str(e) == "'NoneType' object has no attribute 'as_query_engine'":
            return "Please upload a file."
        return f"An error occurred: {e}"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode the images
github_logo_encoded = encode_image("Images/github-logo.png")
linkedin_logo_encoded = encode_image("Images/linkedin-logo.png")
website_logo_encoded = encode_image("Images/ai-logo.png")

# UI Setup
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto Mono")]), css='footer {visibility: hidden}') as demo:
    gr.Markdown("# ChatToFile🤖")
    with gr.Tabs():
        with gr.TabItem("Intro"):
            gr.Markdown(md.description)

        with gr.TabItem("DocBot"):
            with gr.Accordion("=== IMPORTANT: READ ME FIRST ===", open=False):
                guid = gr.Markdown(md.guide)
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(file_count="single", type='filepath', label="Step-1: Upload document")
                    # gr.Markdown("Dont know what to select check out in Intro tab")
                    embed_model_dropdown = gr.Dropdown(embed_models, label="Step-2: Select Embedding", interactive=True)
                    with gr.Row():
                        btn = gr.Button("Submit", variant='primary')
                        clear = gr.ClearButton()
                    output = gr.Text(label='Vector Index')
                    llm_model_dropdown = gr.Dropdown(llm_models, label="Step-3: Select LLM", interactive=True)
                with gr.Column(scale=3):
                    gr.ChatInterface(
                        fn=respond,
                        chatbot=gr.Chatbot(height=500, type='messages'),
                        theme = "soft",
                        type='messages',
                        show_progress='full',
                        # cache_mode='lazy',
                        textbox=gr.Textbox(placeholder="Step-4: Ask me questions on the uploaded document!", container=False)
                    )
    gr.HTML(md.footer.format(github_logo_encoded, linkedin_logo_encoded, website_logo_encoded))
    # Set up Gradio interactions
    llm_model_dropdown.change(fn=set_llm_model, inputs=llm_model_dropdown)
    btn.click(fn=load_files, inputs=[file_input, embed_model_dropdown], outputs=output)
    clear.click(lambda: [None] * 3, outputs=[file_input, embed_model_dropdown, output])

# Launch the demo with a public link option
if __name__ == "__main__":
    demo.launch()