# RAG Financial Documents API

API para consultas em linguagem natural sobre documentos financeiros em PDF. Faça upload de relatórios trimestrais, balanços patrimoniais e demonstrativos de resultado — depois pergunte sobre eles e receba respostas fundamentadas com citação de fonte (documento, página).

## Stack Técnica

- **Python 3.11+** / **FastAPI** — API REST assíncrona
- **LangChain 0.3** — orquestração do pipeline RAG
- **ChromaDB** — banco de dados vetorial persistente
- **OpenAI API** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- **Docling** — extração de texto de PDFs

## Arquitetura

```
PDF Upload
    │
    ▼
[Docling] ──► extrai texto por página
    │
    ▼
[RecursiveCharacterTextSplitter] ──► chunks (1000 chars, overlap 150)
    │
    ▼
[OpenAI Embeddings] ──► vetores 1536-dim
    │
    ▼
[ChromaDB] ──► armazena chunks + metadados

                    Pergunta do usuário
                          │
                          ▼
              [OpenAI Embeddings] ──► vetor da pergunta
                          │
                          ▼
              [ChromaDB] ──► top-K chunks por similaridade de cosseno
                          │
                          ▼
              [gpt-4o-mini] ──► resposta contextualizada + citações
```

## Pré-requisitos

- Python 3.11+
- Conta e chave de API da OpenAI

## Configuração

```bash
git clone <repo>
cd llm-finance

python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt

cp .env.example .env
# edite .env e preencha OPENAI_API_KEY=sk-...
```

## Executando

```bash
uvicorn app.main:app --reload
```

A API estará disponível em `http://localhost:8000`.
Documentação interativa (Swagger): `http://localhost:8000/docs`

## Endpoints

### `POST /documents/upload`

Faz upload e indexação de um documento PDF financeiro.

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@relatorio_trimestral.pdf"
```

Resposta (`201 Created`):
```json
{
  "document_id": "a3f9c1d2e5b7f801",
  "filename": "relatorio_trimestral.pdf",
  "chunk_count": 42,
  "message": "Document 'relatorio_trimestral.pdf' successfully ingested with 42 chunks."
}
```

---

### `GET /documents`

Lista todos os documentos indexados.

```bash
curl http://localhost:8000/documents
```

Resposta (`200 OK`):
```json
{
  "documents": [
    {
      "document_id": "a3f9c1d2e5b7f801",
      "filename": "relatorio_trimestral.pdf",
      "page_count": 15,
      "chunk_count": 42,
      "uploaded_at": "2024-10-01T14:32:00+00:00",
      "size_bytes": 1245678
    }
  ],
  "total": 1
}
```

---

### `POST /query`

Faz uma pergunta em linguagem natural sobre os documentos indexados.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual foi a receita líquida no terceiro trimestre?"}'
```

Resposta (`200 OK`):
```json
{
  "question": "Qual foi a receita líquida no terceiro trimestre?",
  "answer": "A receita líquida no 3T24 foi de R$ 10,2 bilhões [relatorio_trimestral.pdf | page 4].",
  "sources": [
    {
      "document_id": "a3f9c1d2e5b7f801",
      "filename": "relatorio_trimestral.pdf",
      "page_number": 4,
      "chunk_index": 12,
      "relevance_score": 0.9231,
      "excerpt": "A receita líquida consolidada atingiu R$ 10,2 bilhões no 3T24..."
    }
  ],
  "model": "gpt-4o-mini",
  "total_chunks_retrieved": 5
}
```

Parâmetros opcionais:
- `top_k` (int, 1–20): número de trechos a recuperar (padrão: 5)

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | — | **Obrigatório.** Chave da API OpenAI |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Modelo LLM para geração |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Diretório de persistência do ChromaDB |
| `CHROMA_COLLECTION_NAME` | `financial_documents` | Nome da coleção vetorial |
| `CHUNK_SIZE` | `1000` | Tamanho dos chunks em caracteres |
| `CHUNK_OVERLAP` | `150` | Sobreposição entre chunks |
| `RETRIEVAL_TOP_K` | `5` | Número de chunks recuperados por consulta |
| `MAX_UPLOAD_SIZE_MB` | `50` | Tamanho máximo de upload em MB |
| `LOG_LEVEL` | `INFO` | Nível de log |

## Testes

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Estrutura do Projeto

```
llm-finance/
├── app/
│   ├── main.py              # App FastAPI + lifespan
│   ├── config.py            # Configuração via pydantic-settings
│   ├── dependencies.py      # Injeção de dependências FastAPI
│   ├── api/
│   │   ├── documents.py     # POST /documents/upload, GET /documents
│   │   └── query.py         # POST /query
│   ├── schemas/
│   │   ├── documents.py     # Modelos Pydantic de documentos
│   │   └── query.py         # Modelos Pydantic de consulta
│   ├── services/
│   │   ├── ingestion.py     # Pipeline de ingestão de PDFs
│   │   ├── retrieval.py     # Busca vetorial por similaridade
│   │   └── generation.py    # Geração de respostas com LLM
│   ├── core/
│   │   ├── chroma.py        # Cliente e coleções ChromaDB
│   │   ├── embeddings.py    # Wrapper de embeddings OpenAI
│   │   └── exceptions.py    # Exceções de domínio + handlers
│   └── utils/
│       └── pdf.py           # Extração de texto com Docling
└── tests/
    ├── conftest.py          # Fixtures compartilhadas
    ├── test_documents.py    # Testes de integração: endpoints de documentos
    ├── test_query.py        # Testes de integração: endpoint de consulta
    ├── test_ingestion.py    # Testes unitários: IngestionService
    ├── test_retrieval.py    # Testes unitários: RetrievalService
    └── test_generation.py   # Testes unitários: GenerationService
```
