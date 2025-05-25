# 🎓College Handbook RAG System

An advanced Retrieval-Augmented Generation (RAG) system designed to answer questions about college handbook policies and procedures. This system uses a combination of semantic and keyword-based search to provide accurate and contextually relevant answers to student queries.

## Features

- Advanced document processing for PDF, DOCX, and TXT files
- Hybrid search combining dense (semantic) and sparse (keyword) search
- Query enhancement using LLM
- Step-back prompting for better reasoning
- Integration with Google's Gemini AI model
- Robust error handling and logging

## Requirements

- Python 3.8+
- Google API Key for Gemini
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

1. Initialize the RAG system:
```python
from college_rag_system import CollegeHandbookRAG

rag_system = CollegeHandbookRAG()
```

2. Load and index your handbook:
```python
rag_system.load_and_index_handbook("path/to/your/handbook.pdf")
```

3. Ask questions:
```python
response = rag_system.answer_question("What is the attendance policy?")
print(response["answer"])
```

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
