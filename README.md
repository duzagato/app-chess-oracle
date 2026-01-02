# ♟️ Chess Oracle - Árbitro Literário de Xadrez

**Uma LLM especializada que combina literatura clássica de xadrez com análise moderna de engines, executando 100% localmente.**

---

## 📋 Índice

1. [O Problema](#-o-problema)
2. [A Solução](#-a-solução)
3. [O Diferencial](#-o-diferencial)
4. [Arquitetura do Sistema](#-arquitetura-do-sistema)
5. [Requisitos de Sistema](#-requisitos-de-sistema)
6. [Instalação](#-instalação)
7. [Configuração](#-configuração)
8. [Uso Básico](#-uso-básico)
9. [Estrutura do Projeto](#-estrutura-do-projeto)
10. [Roadmap de Desenvolvimento](#-roadmap-de-desenvolvimento)
11. [Especificação Técnica](#-especificação-técnica)

---

## 🎯 O Problema

A literatura clássica de xadrez contém séculos de sabedoria, mas foi escrita em uma era sem engines computacionais. Muitos análises, mesmo de grandes mestres, contêm imprecisões que só podem ser detectadas com ferramentas modernas.

**Desafios:**
- Livros clássicos podem conter análises incorretas
- Engines modernas são precisas, mas não explicam o "porquê"
- Não há uma ferramenta que combine ambos de forma pedagógica

---

## 💡 A Solução

O **Chess Oracle** atua como um "Árbitro Literário" que:

1. **Lê** livros clássicos de xadrez (PDFs)
2. **Consulta** engines modernas para validação técnica
3. **Sintetiza** explicações em linguagem natural que:
   - Validam ou corrigem autores históricos
   - Explicam discrepâncias com empatia pedagógica
   - Contextualizam limitações da época

**Exemplo de output:**
> "Botvinnik recomenda 15.Nf3 nesta posição, sugerindo vantagem branca. No entanto, a Maia Chess identifica que 15.Bxe6! leva a vantagem decisiva (+2.5). O plano de Botvinnik não está errado estrategicamente, mas ele não viu a refutação tática moderna."

---

## 🌟 O Diferencial

### Por que usar Ollama (Execução Local)?

| Aspecto | Ollama Local | APIs Externas (OpenAI, Groq) |
|---------|-------------|------------------------------|
| **Privacidade** | ✅ Dados 100% locais | ❌ Dados enviados para nuvem |
| **Custos** | ✅ Grátis após setup | ❌ Pagamento por token |
| **Rate Limits** | ✅ Sem limites | ❌ Limites de requisições |
| **Offline** | ✅ Funciona sem internet | ❌ Requer conexão |
| **Customização** | ✅ Fine-tuning local futuro | ⚠️ Limitado |

**Desvantagens do Ollama:**
- Requer hardware adequado (RAM, GPU opcional)
- Setup inicial mais complexo
- Velocidade depende do hardware local

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      Chess Oracle                            │
│                    (Orquestrador)                            │
└────┬──────────────────┬─────────────────┬───────────────────┘
     │                  │                 │
     ▼                  ▼                 ▼
┌─────────┐      ┌──────────┐      ┌──────────┐
│ python- │      │ ChromaDB │      │  Ollama  │
│  chess  │      │   +RAG   │      │ (Llama3) │
└─────────┘      └──────────┘      └──────────┘
     │                  │                 │
     ▼                  ▼                 ▼
┌─────────┐      ┌──────────┐      ┌──────────┐
│  Maia   │      │   PDFs   │      │  Local   │
│ Chess   │      │  Livros  │      │   CPU/   │
│ Engine  │      │  Xadrez  │      │   GPU    │
└─────────┘      └──────────┘      └──────────┘
```

### Stack Tecnológica

| Componente | Tecnologia | Função |
|------------|------------|--------|
| **Interface de Xadrez** | `python-chess` | Validação de regras e estados (FEN/PGN) |
| **Cérebro Analítico** | `Maia Chess` | Engine que pensa como humano para detectar erros naturais |
| **Base de Conhecimento** | `ChromaDB` (RAG) | Banco vetorial com livros de xadrez para busca semântica |
| **Embeddings** | `sentence-transformers` | all-MiniLM-L6-v2 para vetorização de textos |
| **Orquestrador (LLM)** | `Ollama (Local)` com `Llama-3` | O "narrador" que traduz dados técnicos em insights - **100% local, sem API externa** |

---

## 💻 Requisitos de Sistema

### Mínimo
- **Sistema Operacional:** Linux, macOS ou Windows
- **RAM:** 8GB (para modelo llama3:8b)
- **CPU:** Processador moderno (Intel i5/AMD Ryzen 5 ou superior)
- **Espaço em Disco:** ~10GB
  - 5GB para modelo Llama-3
  - 2GB para Maia Chess engine
  - 3GB para dependências e dados

### Recomendado
- **RAM:** 16GB ou mais
- **GPU:** NVIDIA com 8GB+ VRAM (para inferência rápida)
- **CPU:** Intel i7/AMD Ryzen 7 ou superior
- **Espaço em Disco:** 20GB+

### Notas sobre Performance
- **Sem GPU:** Inferência em CPU é viável mas mais lenta (3-10s por resposta)
- **Com GPU:** Inferência acelerada (1-3s por resposta)
- **Modelos maiores:** llama3:70b requer 40GB+ RAM

---

## 📦 Instalação

### 1. Instalar Python e Dependências

```bash
# Clone o repositório
git clone https://github.com/duzagato/app-chess-oracle.git
cd app-chess-oracle

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Instalar Ollama

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

#### Windows
Baixe o instalador em: [https://ollama.com/download](https://ollama.com/download)

### 3. Baixar Modelo Llama-3

```bash
# Modelo padrão (8B parâmetros, ~4.7GB)
ollama pull llama3

# Ou modelo maior (70B parâmetros, ~40GB) - apenas se tiver hardware adequado
ollama pull llama3:70b
```

### 4. Baixar Maia Chess Engine

```bash
# Linux/macOS
cd engine/
wget https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1900.elf
chmod +x maia-1900.elf

# Windows
# Baixe manualmente de: https://github.com/CSSLab/maia-chess/releases
# Coloque o binário na pasta engine/
```

**Nota:** Se não tiver a engine Maia, o sistema funcionará apenas sem análise de engine (apenas RAG + LLM).

---

## ⚙️ Configuração

### 1. Copiar Arquivo de Configuração

```bash
cp .env.example .env
```

### 2. Editar Variáveis de Ambiente

Edite o arquivo `.env`:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3

# Maia Chess Engine Path
MAIA_ENGINE_PATH=./engine/maia-1900.elf

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./vector_db

# Chess Books Directory
CHESS_BOOKS_DIRECTORY=./data
```

### 3. Adicionar Livros de Xadrez (Opcional)

Coloque arquivos PDF de livros de xadrez na pasta `data/`:

```bash
cp /caminho/para/seu/livro.pdf data/
```

**Exemplos de livros clássicos:**
- "My System" - Aron Nimzowitsch
- "The Art of Attack in Chess" - Vladimir Vukovic
- "Think Like a Grandmaster" - Alexander Kotov

---

## 🚀 Uso Básico

### 1. Iniciar Servidor Ollama

Em um terminal separado:

```bash
ollama serve
```

### 2. Executar o Chess Oracle

```bash
# Ativar ambiente virtual (se criado)
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Executar exemplo
cd src/
python main.py
```

### 3. Uso Programático

```python
from src.main import ChessOracle

# Inicializa o Oracle
oracle = ChessOracle()

# Analisa uma posição
fen = "rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4"
query = "Quais são os planos para as pretas nesta Defesa Siciliana?"

result = oracle.analyze(fen, query)

print(result['oracle_response'])

# Fecha recursos
oracle.close()
```

### 4. Carregar Livros de Xadrez (RAG)

```python
from src.rag_engine import ChessRAGEngine

# Inicializa RAG
rag = ChessRAGEngine()

# Carrega todos os PDFs da pasta data/
results = rag.load_directory('./data')

print(f"Total de chunks adicionados: {sum(results.values())}")
```

---

## 📁 Estrutura do Projeto

```
chess-ai-project/
├── data/                      # PDFs e livros de xadrez
│   └── .gitkeep
├── vector_db/                 # Persistência do ChromaDB
│   └── .gitkeep
├── engine/                    # Binários da Maia Chess
│   └── .gitkeep
├── src/
│   ├── main.py               # Entry point do MVP
│   ├── chess_logic.py        # Integração com python-chess e Maia
│   ├── rag_engine.py         # Lógica de busca no banco vetorial
│   ├── llm_engine.py         # Integração com Ollama local
│   └── prompt_template.py    # Definição das instruções da LLM
├── requirements.txt          # Dependências Python
├── .env.example              # Exemplo de configuração
├── .gitignore               # Arquivos ignorados pelo Git
└── README.md                # Este arquivo
```

---

## 🗺️ Roadmap de Desenvolvimento

### Fase 1: MVP Funcional ✅
- [x] Estrutura base do projeto
- [x] Integração com python-chess
- [x] Integração com Ollama local
- [x] Sistema RAG com ChromaDB
- [x] Orquestração básica
- [x] Documentação inicial

### Fase 2: Refinamento (Next)
- [ ] Interface CLI interativa
- [ ] Sistema de cache de análises
- [ ] Suporte a múltiplos modelos Ollama
- [ ] Testes automatizados
- [ ] Performance profiling
- [ ] Logging estruturado

### Fase 3: Features Avançadas
- [ ] Interface Web (Gradio/Streamlit)
- [ ] Análise de partidas completas (PGN)
- [ ] Comparação side-by-side (livro vs engine)
- [ ] Exportação de relatórios
- [ ] Fine-tuning local do modelo
- [ ] Suporte a múltiplas engines

### Fase 4: Produção
- [ ] API REST
- [ ] Sistema de usuários
- [ ] Dashboard de estatísticas
- [ ] Mobile app (React Native)
- [ ] Integração com chess.com/lichess

---

## 🔧 Especificação Técnica

### Fluxo de Trabalho

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Usuário fornece FEN + Query                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ 2. ChessOracle valida FEN (python-chess)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐  ┌─────────────────────┐
│ 3a. RAG Engine   │  │ 3b. Chess Analyzer  │
│  - Busca query   │  │  - Analisa posição  │
│  - Retorna top-k │  │  - Retorna eval +   │
│    documentos    │  │    melhor lance     │
└────────┬─────────┘  └──────────┬──────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Monta prompt com:                                         │
│    - FEN da posição                                         │
│    - Query do usuário                                       │
│    - Contexto dos livros (RAG)                              │
│    - Avaliação da engine                                    │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Ollama LLM gera resposta sintetizada                     │
│    - System prompt: "Você é o Chess Oracle..."             │
│    - Compara literatura histórica vs análise moderna       │
│    - Explica discrepâncias com empatia pedagógica          │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Retorna análise completa ao usuário                      │
└─────────────────────────────────────────────────────────────┘
```

### Módulos Detalhados

#### `chess_logic.py`
- **Classe:** `ChessAnalyzer`
- **Funções:**
  - Validação de FEN/PGN
  - Comunicação UCI com Maia Engine
  - Análise de posições (avaliação + melhor lance)
  - Informações de posição (xeque, xeque-mate, etc.)

#### `rag_engine.py`
- **Classe:** `ChessRAGEngine`
- **Funções:**
  - Ingestão de PDFs (PyPDFLoader)
  - Chunking de texto (RecursiveCharacterTextSplitter)
  - Geração de embeddings (sentence-transformers)
  - Busca semântica (ChromaDB)
  - Persistência de banco vetorial

#### `llm_engine.py`
- **Classe:** `OllamaLLM`
- **Funções:**
  - Health check do servidor Ollama
  - Listagem de modelos disponíveis
  - Geração de texto (generate)
  - Chat multi-turn (chat)
  - Pull de modelos

#### `prompt_template.py`
- **Classe:** `ChessOraclePrompts`
- **Funções:**
  - System prompt (definição do papel)
  - Analysis prompt (análise completa)
  - Comparison prompt (literatura vs engine)
  - Position explanation (explicação geral)
  - Move explanation (explicação de lance)

#### `main.py`
- **Classe:** `ChessOracle`
- **Funções:**
  - Inicialização de todos os componentes
  - Orquestração do fluxo de análise
  - Fallback quando componentes não disponíveis
  - Exemplo de uso

---

## 🐛 Troubleshooting

### Ollama não está rodando
```bash
# Verificar se está rodando
curl http://localhost:11434

# Se não retornar nada, inicie o servidor
ollama serve
```

### Modelo não encontrado
```bash
# Listar modelos instalados
ollama list

# Baixar modelo
ollama pull llama3
```

### Engine Maia não encontrada
- Verifique se o binário está em `engine/maia-1900.elf`
- Verifique permissões de execução: `chmod +x engine/maia-1900.elf`
- O sistema funciona sem engine (apenas RAG + LLM)

### ChromaDB errors
```bash
# Limpar banco vetorial
rm -rf vector_db/*
```

### Importação de módulos falha
```bash
# Certifique-se de estar no diretório correto
cd src/

# Ou adicione ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/caminho/para/app-chess-oracle/src"
```

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📧 Contato

Para dúvidas, sugestões ou reportar bugs, abra uma issue no GitHub.

---

## 🙏 Agradecimentos

- **Maia Chess Team** - pela engine que pensa como humano
- **Ollama Team** - por tornar LLMs locais acessíveis
- **python-chess** - biblioteca fundamental para o projeto
- **ChromaDB** - banco vetorial simples e eficiente
- **LangChain** - ferramentas para RAG

---

**♟️ Chess Oracle - Onde a tradição encontra a inovação ♟️**