# Local RAG with Hugging Face

A lightweight Retrieval-Augmented Generation system for asking questions to PDF documents built using:

- Hugging Face Transformers
- Sentence Transformers
- FAISS

## Installation

### Requirements

* Python 3.11 or later
* Conda
* A Hugging Face account is optional, but may be required for some models

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create the Conda environment

Create the environment from the included `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate doc-chat
```

### 3. Verify the installation

Run the test suite:

```bash
pytest
```

If the tests pass, the installation is complete.

## How to use

Create a the directories `data/documents/`. Place the PDF document inside:

For example:

```text
data/
└── documents/
    └── my_pdf.pdf
```

Index the document by:

```bash
python -m doc_chat index data/documents/my_pdf.pdf
```

Ask questions about the document by:

```bash
python -m doc_chat chat --index my_pdf
```

## Limitations
Metadata questions, such as asking for a paper’s title or authors, may perform worse than semantic content questions because the current system relies on vector retrieval over extracted text.