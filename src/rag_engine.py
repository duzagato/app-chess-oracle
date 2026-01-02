"""
Módulo RAG (Retrieval-Augmented Generation) para busca semântica em livros de xadrez.

Este módulo fornece funcionalidades para:
- Ingestão de PDFs de livros de xadrez
- Criação e gerenciamento de banco vetorial com ChromaDB
- Busca semântica de contexto relevante usando embeddings
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


class ChessRAGEngine:
    """
    Engine de busca semântica para literatura de xadrez usando RAG.
    
    Utiliza ChromaDB para persistência vetorial e HuggingFace embeddings
    para representação semântica dos textos.
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "chess_books"
    ):
        """
        Inicializa o RAG Engine.
        
        Args:
            persist_dir: Diretório para persistência do ChromaDB
            embedding_model: Nome do modelo de embeddings
            collection_name: Nome da coleção no ChromaDB
        """
        self.persist_dir = persist_dir or os.getenv(
            'CHROMADB_PERSIST_DIR', 
            './vector_db'
        )
        self.embedding_model_name = embedding_model or os.getenv(
            'EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.collection_name = collection_name
        
        # Inicializa componentes
        self._initialize_embeddings()
        self._initialize_chromadb()
    
    def _initialize_embeddings(self) -> None:
        """Inicializa o modelo de embeddings."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Usa CPU por padrão
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _initialize_chromadb(self) -> None:
        """Inicializa o cliente ChromaDB e a coleção."""
        # Cria diretório se não existir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Inicializa cliente ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Obtém ou cria coleção
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Chess books vector database"}
            )
    
    def ingest_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> int:
        """
        Ingere um PDF de livro de xadrez no banco vetorial.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            chunk_size: Tamanho dos chunks de texto (padrão: 1000)
            chunk_overlap: Sobreposição entre chunks (padrão: 200)
            
        Returns:
            Número de chunks adicionados ao banco
        """
        # Verifica se o arquivo existe
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
        
        # Carrega o PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Divide em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Prepara dados para o ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                'source': pdf_path,
                'page': chunk.metadata.get('page', 0),
                'chunk_id': i
            }
            for i, chunk in enumerate(chunks)
        ]
        ids = [
            f"{Path(pdf_path).stem}_chunk_{i}" 
            for i in range(len(chunks))
        ]
        
        # Gera embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Adiciona ao ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def ingest_directory(
        self,
        data_dir: Optional[str] = None,
        file_pattern: str = "*.pdf"
    ) -> Dict[str, int]:
        """
        Ingere todos os PDFs de um diretório.
        
        Args:
            data_dir: Diretório contendo PDFs (padrão: ./data)
            file_pattern: Padrão de arquivos (padrão: *.pdf)
            
        Returns:
            Dict mapeando nome do arquivo para número de chunks
        """
        data_dir = data_dir or os.getenv('DATA_DIR', './data')
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")
        
        results = {}
        pdf_files = list(data_path.glob(file_pattern))
        
        if not pdf_files:
            print(f"Nenhum arquivo {file_pattern} encontrado em {data_dir}")
            return results
        
        for pdf_file in pdf_files:
            try:
                print(f"Processando: {pdf_file.name}...")
                num_chunks = self.ingest_pdf(str(pdf_file))
                results[pdf_file.name] = num_chunks
                print(f"  ✓ {num_chunks} chunks adicionados")
            except Exception as e:
                print(f"  ✗ Erro ao processar {pdf_file.name}: {e}")
                results[pdf_file.name] = 0
        
        return results
    
    def search_context(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Busca contexto relevante no banco vetorial.
        
        Args:
            query: Consulta em linguagem natural
            k: Número de resultados a retornar (padrão: 3)
            min_score: Score mínimo de similaridade (padrão: 0.0)
            
        Returns:
            Lista de dicts contendo:
                - 'text': Texto do chunk
                - 'source': Arquivo de origem
                - 'page': Número da página
                - 'score': Score de similaridade
        """
        # Gera embedding da query
        query_embedding = self.embeddings.embed_query(query)
        
        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Formata resultados
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                # ChromaDB retorna distâncias (menor = mais similar)
                # Converte para score de similaridade
                distance = results['distances'][0][i]
                # Protege contra valores negativos ou muito pequenos
                similarity_score = 1.0 / (1.0 + max(0.0, distance))
                
                if similarity_score >= min_score:
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'source': results['metadatas'][0][i].get('source', 'unknown'),
                        'page': results['metadatas'][0][i].get('page', 0),
                        'score': round(similarity_score, 4)
                    })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """
        Retorna estatísticas da coleção.
        
        Returns:
            Dict com número de documentos e outros metadados
        """
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_dir': self.persist_dir,
            'embedding_model': self.embedding_model_name
        }
    
    def reset_collection(self) -> None:
        """
        Remove todos os documentos da coleção.
        
        ATENÇÃO: Esta operação é irreversível!
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Chess books vector database"}
        )
        print(f"Coleção '{self.collection_name}' resetada com sucesso.")


# Exemplo de uso
if __name__ == "__main__":
    # Inicializa o RAG Engine
    rag = ChessRAGEngine()
    
    # Exibe estatísticas
    stats = rag.get_collection_stats()
    print(f"Estatísticas da coleção: {stats}")
    
    # Exemplo de busca
    query = "Como jogar a abertura Ruy Lopez?"
    print(f"\nBuscando contexto para: '{query}'")
    results = rag.search_context(query, k=2)
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n--- Resultado {i} (Score: {result['score']}) ---")
            print(f"Fonte: {Path(result['source']).name}, Página: {result['page']}")
            print(f"Texto: {result['text'][:200]}...")
    else:
        print("Nenhum contexto encontrado. Adicione PDFs ao diretório data/ primeiro.")
        print("\nPara adicionar livros:")
        print("  rag.ingest_pdf('caminho/para/livro.pdf')")
        print("  ou")
        print("  rag.ingest_directory('./data')")
