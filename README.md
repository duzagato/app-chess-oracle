# ♟️ Chess Oracle - Árbitro Literário de Xadrez

> *Sincronizando a sabedoria clássica do xadrez com a precisão da IA moderna*

Chess Oracle é uma LLM especializada que atua como um "Árbitro Literário" para xadrez, combinando análise de engines modernas (Maia Chess) com literatura clássica de xadrez. O sistema lê livros históricos, consulta engines de ponta, e gera explicações em linguagem natural que validam ou corrigem autores clássicos.

---

## 📋 Índice

- [O Problema](#-o-problema)
- [A Solução](#-a-solução)
- [O Diferencial](#-o-diferencial)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Fluxo de Trabalho](#-fluxo-de-trabalho)
- [Instalação](#-instalação)
- [Uso Básico](#-uso-básico)
- [Roadmap de Desenvolvimento](#-roadmap-de-desenvolvimento)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Especificação Técnica](#-especificação-técnica)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

---

## 🎯 O Problema

Entusiastas e jogadores de xadrez frequentemente estudam livros clássicos escritos há décadas. Embora essas obras sejam ricas em conceitos estratégicos, muitas análises contêm **erros táticos** que só foram descobertos com o advento das engines modernas. 

**Desafios atuais:**
- Livros clássicos podem conter análises refutadas por computadores
- Motores de xadrez modernos fornecem avaliações precisas, mas carecem de explicações educativas
- Não existe uma ponte entre a literatura histórica e a análise computacional moderna
- Aprender xadrez requer tempo para confrontar teoria clássica com análise moderna

## 💡 A Solução

O **Chess Oracle** é um sistema de IA que:

1. **Lê e Indexa** livros de xadrez clássicos usando RAG (Retrieval-Augmented Generation)
2. **Analisa Posições** com a Maia Chess, uma engine que pensa como humanos
3. **Compara e Explica** diferenças entre o comentário histórico e a realidade técnica atual
4. **Educa com Empatia** - explica por que autores históricos erraram, mantendo respeito por suas intenções estratégicas

### Exemplo de Uso

```python
from src.main import ChessOracle

# Inicializa o Oracle
oracle = ChessOracle()

# Carrega livros de xadrez
oracle.load_books('./data')

# Analisa uma posição da Ruy Lopez
fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
query = "Por que Bb5 é considerado o movimento mais forte aqui?"

result = oracle.analyze(fen=fen, user_query=query)
print(result['response'])
```

**Resposta do Oracle:**
> *"A Abertura Ruy Lopez (3.Bb5) é considerada uma das aberturas mais sólidas das brancas. Segundo Capablanca em 'Chess Fundamentals', o bispo em b5 pressiona o cavalo em c6, que defende o peão central e5. A análise moderna da Maia confirma essa avaliação com +0.3, indicando ligeira vantagem das brancas. O autor clássico acertou na avaliação estratégica: controle do centro e desenvolvimento harmonioso..."*

## 🌟 O Diferencial

| Característica | Chess Oracle | Engines Tradicionais | Livros Clássicos |
|---------------|--------------|---------------------|------------------|
| **Análise Técnica Precisa** | ✅ | ✅ | ❌ |
| **Explicações Educativas** | ✅ | ❌ | ✅ |
| **Contexto Histórico** | ✅ | ❌ | ✅ |
| **Validação de Literatura** | ✅ | ❌ | ❌ |
| **Linguagem Natural** | ✅ | ❌ | ✅ |
| **Atualização Contínua** | ✅ | ✅ | ❌ |

---

## 🏗 Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      CHESS ORACLE                            │
│                  (Orchestrador Principal)                    │
└───────┬────────────────────┬───────────────────┬────────────┘
        │                    │                   │
        ▼                    ▼                   ▼
┌───────────────┐    ┌──────────────┐    ┌─────────────────┐
│   RAG Engine  │    │ Chess Logic  │    │ LLM (Llama-3)   │
│   (ChromaDB)  │    │ (Maia Chess) │    │  via Groq API   │
└───────┬───────┘    └──────┬───────┘    └─────────┬───────┘
        │                   │                       │
        ▼                   ▼                       ▼
┌───────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Livros PDFs   │    │ python-chess │    │ Prompt Template │
│ (data/)       │    │ Engine UCI   │    │ (Sistema)       │
└───────────────┘    └──────────────┘    └─────────────────┘
```

### Stack Tecnológica

| Componente | Tecnologia | Função |
|------------|------------|--------|
| **Interface de Xadrez** | `python-chess` | Validação de regras e estados (FEN/PGN) |
| **Cérebro Analítico** | `Maia Chess` | Engine que pensa como humano para detectar erros naturais |
| **Base de Conhecimento** | `ChromaDB` (RAG) | Banco vetorial com livros de xadrez para busca semântica |
| **Embeddings** | `sentence-transformers` | Modelo all-MiniLM-L6-v2 para vetorização de texto |
| **Orquestrador (LLM)** | `Llama-3 (Groq API)` | O "narrador" que traduz dados técnicos em insights |
| **Loader de PDFs** | `pypdf` | Extração de texto de livros em PDF |
| **Framework RAG** | `langchain` | Orquestração de pipelines RAG |

---

## 🔄 Fluxo de Trabalho

```
1️⃣ USUÁRIO
   │
   ├─> Fornece FEN da posição
   └─> Faz uma pergunta sobre a posição
          │
          ▼
2️⃣ RAG ENGINE (ChromaDB)
   │
   ├─> Busca semântica nos livros clássicos
   ├─> Retorna top-3 trechos relevantes
   └─> Contexto histórico extraído
          │
          ▼
3️⃣ CHESS ANALYZER (Maia)
   │
   ├─> Valida FEN
   ├─> Analisa posição via UCI
   └─> Retorna: avaliação + melhor movimento
          │
          ▼
4️⃣ PROMPT BUILDER
   │
   ├─> Combina contexto histórico + análise moderna
   ├─> Formata prompt de comparação
   └─> Adiciona instruções de "árbitro literário"
          │
          ▼
5️⃣ LLM (Llama-3 via Groq)
   │
   ├─> Recebe prompt estruturado
   ├─> Compara literatura com análise
   ├─> Gera explicação empática
   └─> Retorna resposta em linguagem natural
          │
          ▼
6️⃣ RESPOSTA FINAL
   │
   └─> Síntese educativa que valida ou corrige
       autores clássicos com contexto histórico
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- Conta na [Groq API](https://console.groq.com/) (para LLM)
- Maia Chess Engine ([Download](https://github.com/CSSLab/maia-chess))

### Passo a Passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/duzagato/app-chess-oracle.git
   cd app-chess-oracle
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure variáveis de ambiente**
   ```bash
   cp .env.example .env
   # Edite .env e adicione sua GROQ_API_KEY
   ```

4. **Baixe a Maia Chess Engine**
   - Baixe o binário da [Maia Chess](https://github.com/CSSLab/maia-chess)
   - Coloque o executável em `./engine/maia`
   - Dê permissão de execução: `chmod +x ./engine/maia`

5. **Adicione livros de xadrez**
   - Coloque arquivos PDF de livros de xadrez em `./data/`
   - Exemplos: "My System" (Nimzowitsch), "Modern Chess Openings", etc.

6. **Execute o MVP**
   ```bash
   python src/main.py
   ```

---

## 📖 Uso Básico

### 1. Inicialização

```python
from src.main import ChessOracle

# Inicializa o sistema
oracle = ChessOracle()

# Verifica estatísticas
stats = oracle.get_stats()
print(f"Chunks na base: {stats['rag']['total_chunks']}")
```

### 2. Carregar Livros

```python
# Carrega todos os PDFs do diretório data/
results = oracle.load_books('./data')

for book, chunks in results.items():
    print(f"✓ {book}: {chunks} chunks processados")
```

### 3. Analisar Posição

```python
# FEN de uma posição interessante
fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

# Faz uma pergunta
query = "Por que Bb5 é o melhor movimento para as brancas?"

# Obtém análise
result = oracle.analyze(
    fen=fen,
    user_query=query,
    engine_depth=20,  # Profundidade da engine
    rag_k=3,          # Top-3 contextos
    engine_time=1.0   # 1 segundo de análise
)

if result['success']:
    print(result['response'])
else:
    print(f"Erro: {result['error']}")
```

### 4. Usando Componentes Isoladamente

```python
# Apenas busca RAG
from src.rag_engine import ChessRAGEngine

rag = ChessRAGEngine()
contexts = rag.search_context("Defesa Siciliana", k=5)

# Apenas análise da engine
from src.chess_logic import ChessAnalyzer

analyzer = ChessAnalyzer()
analysis = analyzer.analyze_position(fen, depth=20)
analyzer.close_engine()
```

---

## 🗺 Roadmap de Desenvolvimento

### ✅ Fase 1: MVP Funcional (Atual)
- [x] Estrutura básica do projeto
- [x] Integração com python-chess
- [x] RAG Engine com ChromaDB
- [x] Integração com Maia Chess
- [x] Orquestrador com Groq/Llama-3
- [x] Documentação completa

### 🔄 Fase 2: Aprimoramentos (Q1 2026)
- [ ] Interface web com Streamlit/Gradio
- [ ] Suporte a múltiplas engines (Stockfish, Leela)
- [ ] Análise de partidas completas (PGN)
- [ ] Sistema de cache para análises
- [ ] Melhorias no chunking de PDFs
- [ ] Suporte a OCR para livros escaneados

### 🔮 Fase 3: Recursos Avançados (Q2 2026)
- [ ] Fine-tuning do LLM em literatura de xadrez
- [ ] Sistema de rankings de fontes
- [ ] API REST para integração externa
- [ ] Dashboard de métricas e analytics
- [ ] Suporte multiidioma
- [ ] Mobile app (iOS/Android)

---

## 📁 Estrutura do Projeto

```
chess-oracle/
├── data/                   # PDFs de livros de xadrez
│   ├── .gitkeep
│   └── [seus_livros.pdf]
│
├── vector_db/              # Persistência do ChromaDB
│   ├── .gitkeep
│   └── [arquivos_chromadb]
│
├── engine/                 # Binários da Maia Chess
│   ├── .gitkeep
│   └── maia               # Engine executável
│
├── src/                    # Código fonte
│   ├── main.py            # Entry point do MVP
│   ├── rag_engine.py      # Lógica RAG com ChromaDB
│   ├── chess_logic.py     # Integração python-chess + Maia
│   └── prompt_template.py # Templates de prompt para LLM
│
├── .env.example           # Template de variáveis de ambiente
├── .gitignore            # Arquivos ignorados pelo Git
├── requirements.txt      # Dependências Python
└── README.md            # Este arquivo
```

---

## 🔧 Especificação Técnica

### Requisitos de Sistema

- **Python**: 3.8 ou superior
- **Memória RAM**: 4GB mínimo (8GB recomendado)
- **Espaço em Disco**: 2GB (para embeddings e base vetorial)
- **CPU**: Multi-core recomendado para análise de engine

### Dependências Principais

```
python-chess>=1.999          # Lógica de xadrez e UCI
langchain>=0.1.0            # Framework RAG
langchain-community>=0.0.10 # Loaders e integrações
chromadb>=0.4.0             # Banco vetorial
sentence-transformers>=2.2.0 # Embeddings
groq>=0.4.0                 # Cliente Groq API
pypdf>=3.0.0                # Extração de PDF
python-dotenv>=1.0.0        # Gerenciamento de .env
```

### Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `GROQ_API_KEY` | API key do Groq | - (obrigatório) |
| `MAIA_ENGINE_PATH` | Caminho para Maia | `./engine/maia` |
| `CHROMADB_PERSIST_DIR` | Diretório ChromaDB | `./vector_db` |
| `DATA_DIR` | Diretório de PDFs | `./data` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_MODEL` | Modelo LLM | `llama-3.1-70b-versatile` |

### Formato FEN

O Chess Oracle aceita posições no formato **FEN (Forsyth-Edwards Notation)**:

```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

- **Posição inicial**: Tabuleiro 8x8
- **Turno**: w (brancas) ou b (pretas)
- **Roque**: KQkq (disponibilidade)
- **En passant**: Coluna disponível ou -
- **Meios movimentos**: Contagem para regra dos 50 movimentos
- **Movimentos completos**: Número do movimento

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Áreas de Contribuição

- 📚 **Curadoria de Livros**: Adicionar mais obras clássicas
- 🐛 **Bug Fixes**: Reportar e corrigir problemas
- ✨ **Novas Features**: Implementar itens do roadmap
- 📖 **Documentação**: Melhorar docs e exemplos
- 🧪 **Testes**: Adicionar cobertura de testes

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

## 🙏 Agradecimentos

- **Maia Chess Team**: Por criar uma engine que pensa como humanos
- **Groq**: Por fornecer acesso à API Llama-3
- **ChromaDB**: Por um banco vetorial eficiente
- **python-chess**: Por uma biblioteca robusta de xadrez
- **Mestres Clássicos**: Capablanca, Nimzowitsch, Alekhine e todos que contribuíram para a literatura de xadrez

---

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/duzagato/app-chess-oracle/issues)
- **Discussões**: [GitHub Discussions](https://github.com/duzagato/app-chess-oracle/discussions)

---

<div align="center">

**♟️ Chess Oracle - Onde a História encontra o Futuro do Xadrez ♟️**

⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!

</div>