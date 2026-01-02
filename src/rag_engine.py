"""
Módulo RAG (Retrieval-Augmented Generation) para busca semântica em livros de xadrez.

Este módulo fornece a classe ChessRAGEngine que usa ChromaDB para armazenar
e buscar contexto relevante de livros de xadrez usando embeddings.
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class ChessRAGEngine:
    """
    Engine RAG para busca semântica em literatura de xadrez.
    
    Utiliza ChromaDB para armazenamento vetorial e sentence-transformers
    para geração de embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = "chess_books",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Inicializa o engine RAG.
        
        Args:
            persist_directory: Diretório para persistência do ChromaDB
            collection_name: Nome da coleção no ChromaDB
            embedding_model: Modelo de embeddings do sentence-transformers
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Inicializa modelo de embeddings
        print(f"Carregando modelo de embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Inicializa ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Obtém ou cria coleção
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Coleção '{collection_name}' carregada: {self.collection.count()} documentos")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"✓ Coleção '{collection_name}' criada")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos para gerar embeddings
        
        Returns:
            Lista de embeddings (vetores de floats)
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def load_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """
        Carrega um PDF e adiciona ao banco vetorial.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
        
        Returns:
            Número de chunks adicionados
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        
        print(f"\nCarregando PDF: {pdf_path}")
        
        # Carrega o PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"  {len(documents)} páginas carregadas")
        
        # Divide em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"  {len(chunks)} chunks criados")
        
        # Prepara dados para o ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        
        # Gera IDs únicos
        filename = os.path.basename(pdf_path)
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        
        # Gera metadados
        metadatas = [
            {
                "source": pdf_path,
                "filename": filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Gera embeddings
        print(f"  Gerando embeddings...")
        embeddings = self._generate_embeddings(texts)
        
        # Adiciona ao ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ {len(chunks)} chunks adicionados à coleção")
        return len(chunks)
    
    def load_directory(
        self,
        directory_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, int]:
        """
        Carrega todos os PDFs de um diretório.
        
        Args:
            directory_path: Caminho do diretório com PDFs
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
        
        Returns:
            Dicionário com nome do arquivo e número de chunks adicionados
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        
        results = {}
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️ Nenhum arquivo PDF encontrado em: {directory_path}")
            return results
        
        print(f"\n{'='*60}")
        print(f"Carregando {len(pdf_files)} PDFs de: {directory_path}")
        print(f"{'='*60}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            try:
                count = self.load_pdf(pdf_path, chunk_size, chunk_overlap)
                results[pdf_file] = count
            except Exception as e:
                print(f"⚠️ Erro ao carregar {pdf_file}: {e}")
                results[pdf_file] = 0
        
        print(f"\n{'='*60}")
        print(f"Total de chunks adicionados: {sum(results.values())}")
        print(f"{'='*60}\n")
        
        return results
    
    def search_context(self, query: str, k: int = 3) -> List[Dict]:
        """
        Busca contexto relevante no banco vetorial.
        
        Args:
            query: Query de busca
            k: Número de resultados a retornar
        
        Returns:
            Lista de dicionários com resultados:
            - 'text': Texto do chunk
            - 'metadata': Metadados (source, page, etc)
            - 'distance': Distância no espaço vetorial (menor = mais similar)
        """
        # Gera embedding da query
        query_embedding = self._generate_embeddings([query])[0]
        
        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Formata resultados
        formatted_results = []
        
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """
        Retorna estatísticas da coleção.
        
        Returns:
            Dicionário com estatísticas:
            - 'total_documents': Total de documentos
            - 'collection_name': Nome da coleção
        """
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection_name
        }
    
    def clear_collection(self):
        """
        Remove todos os documentos da coleção.
        
        ATENÇÃO: Esta operação não pode ser desfeita!
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"✓ Coleção '{self.collection_name}' limpa")
        except Exception as e:
            print(f"⚠️ Erro ao limpar coleção: {e}")


if __name__ == "__main__":
    # Exemplo de uso
    print("=== Chess Oracle - RAG Engine ===\n")
    
    # Inicializa o RAG engine
    rag = ChessRAGEngine(
        persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', './vector_db')
    )
    
    # Verifica estatísticas
    stats = rag.get_collection_stats()
    print(f"Coleção: {stats['collection_name']}")
    print(f"Documentos: {stats['total_documents']}\n")
    
    # Tenta carregar PDFs se houver
    books_dir = os.getenv('CHESS_BOOKS_DIRECTORY', './data')
    
    if os.path.exists(books_dir):
        pdf_files = [f for f in os.listdir(books_dir) if f.endswith('.pdf')]
        
        if pdf_files:
            print(f"Encontrados {len(pdf_files)} PDFs em {books_dir}")
            
            # Pergunta se deve carregar (apenas em modo interativo)
            if stats['total_documents'] == 0:
                print("Base de dados vazia. Carregue PDFs manualmente com:")
                print(f"  rag.load_directory('{books_dir}')")
        else:
            print(f"⚠️ Nenhum PDF encontrado em {books_dir}")
            print("   Adicione livros de xadrez em PDF para usar o RAG\n")
    
    # Exemplo de busca (se houver documentos)
    if stats['total_documents'] > 0:
        print("\nExemplo de busca:")
        query = "Defesa Siciliana"
        results = rag.search_context(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Fonte: {result['metadata']['filename']}")
            print(f"   Página: {result['metadata']['page']}")
            print(f"   Texto: {result['text'][:200]}...")
