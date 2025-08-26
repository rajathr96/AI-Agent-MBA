📚 MBA AI Document Agent
A Streamlit-powered application that allows users to query their local MBA documents. This agent leverages Retrieval-Augmented Generation (RAG) to provide answers and pinpoint relevant sources directly from your study materials, ensuring responses are grounded in your specific knowledge base.

✨ Features
Local Document Processing: Scans a user-specified local folder (and its subdirectories) for PDF and plain text (.txt) files.

Intelligent Answering: Uses Google's Gemini-1.5-Flash model to answer questions based on the content of your documents.

Source Attribution: Provides the specific document(s) and page numbers used to formulate an answer, with an option to view the raw text.

User-Friendly Interface: Built with Streamlit for an interactive web application experience.

Dynamic Path Input: Users can easily specify the local directory containing their MBA materials directly within the UI.

Optimized Performance: Utilizes nest_asyncio for stable asynchronous operations and st.session_state to prevent redundant document processing on app reruns.

💡 How It Works (Retrieval-Augmented Generation - RAG)
This application employs a Retrieval-Augmented Generation (RAG) approach:

Document Loading & Chunking: When you provide a folder path, the application loads your PDF and text documents, then splits them into smaller, manageable chunks.

Embedding Generation: Each text chunk is converted into a high-dimensional numerical vector (an "embedding") using Google's embedding-001 model. These embeddings capture the semantic meaning of the text.

Vector Database (FAISS): These embeddings are stored in a local FAISS (Facebook AI Similarity Search) vector database, enabling ultra-fast similarity searches.

Question Answering: When you ask a question, your query is also converted into an embedding. The FAISS database quickly retrieves the most semantically relevant text chunks from your MBA documents.

Generative AI: These retrieved chunks are then passed as context to the Google Gemini-1.5-Flash Large Language Model (LLM). The LLM uses this specific context, along with its vast general knowledge, to generate a precise answer, explicitly instructed to respond only from the provided information.

This ensures that the answers are accurate and directly attributable to your personal study materials.

⚙️ Prerequisites
Before you begin, ensure you have the following installed:

Python 3.9+ (or a compatible version)

All necessary Python libraries (e.g., Streamlit, LangChain, pypdf, faiss-cpu, google-generativeai, nest-asyncio, etc.) installed globally or in your active environment.

Git (for cloning the repository)

🚀 Setup and Installation
Follow these steps to get your MBA AI Document Agent up and running on your local machine.

1. Clone the Repository
First, clone this GitHub repository to your local machine:

git clone https://github.com/rajathr96/AI-Agent-MBA.git
cd AI-Agent-MBA

2. Obtain a Google Gemini API Key
This application relies on the Google Gemini API for generating embeddings and answers.

Go to Google AI Studio.

Sign in with your Google account.

Click "Create API key in new project" or "Get API key".

Copy your generated API key.

3. Configure Your API Key
Open the gemini.py file in a text editor and locate the GOOGLE_API_KEY variable:

# --- Configuration ---
# Your Google Gemini API key as provided
GOOGLE_API_KEY = "YOUR_API_KEY_HERE" # Replace "YOUR_API_KEY_HERE" with your actual Gemini API key

Replace "YOUR_API_KEY_HERE" with the API key you obtained from Google AI Studio.

▶️ Running the Application
Once everything is set up, you can run the Streamlit application:

python3 -m streamlit run gemini.py

This command will open the Streamlit application in your default web browser (usually at http://localhost:8501).

📝 Usage
Enter Document Path: In the Streamlit UI, you'll see a text input field labeled "1. Specify Your MBA Documents Folder". Enter the full local path to the directory containing your MBA PDF and/or .txt files (e.g., /Users/yourname/Documents/MBA_Materials). Press Enter or click outside the box.

Knowledge Base Building: The application will then display a spinner indicating that it's "Building Your Knowledge Base." This involves loading, chunking, embedding, and indexing your documents. This step can take some time depending on the number and size of your files.

Ask Questions: Once the knowledge base is ready, a text area labeled "3. Ask a Question about your MBA Documents" will appear. Type your question and press Enter.

View Answer & Sources:

The answer will be displayed in the "Answer" tab.

Click the "Sources" tab to see the specific documents and pages the AI used to formulate its response, with expandable sections to view the exact content.

📂 Project Structure
AI-Agent-MBA/
├── gemini.py             # The main Streamlit application script
└── README.md             # This file

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, bug reports, or new features, please open an issue or submit a pull request on the GitHub repository.

📄 License
This project is open-source and available under the MIT License. See the LICENSE file for more details.