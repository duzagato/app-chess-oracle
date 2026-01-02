"""
Módulo de integração com Ollama para execução local de LLMs.

Este módulo fornece a classe OllamaLLM para interagir com modelos de linguagem
executando localmente via Ollama, sem necessidade de APIs externas.
"""

import os
import requests
from typing import Dict, List, Optional
import json


class OllamaLLM:
    """
    Cliente para integração com Ollama (execução local de LLMs).
    
    Permite executar modelos como Llama-3 localmente sem APIs externas,
    garantindo privacidade e eliminando custos de API.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.7,
        timeout: int = 120
    ):
        """
        Inicializa o cliente Ollama.
        
        Args:
            host: URL do servidor Ollama (default: http://localhost:11434)
            model: Nome do modelo a usar (default: llama3)
            temperature: Temperatura para geração (0.0-1.0)
            timeout: Timeout em segundos para requisições
        """
        self.host = host.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_base = f"{self.host}/api"
    
    def is_running(self) -> bool:
        """
        Verifica se o servidor Ollama está rodando.
        
        Returns:
            True se o servidor está acessível
        """
        try:
            response = requests.get(f"{self.host}/", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[Dict]:
        """
        Lista todos os modelos disponíveis no Ollama.
        
        Returns:
            Lista de dicionários com informações dos modelos
        """
        if not self.is_running():
            return []
        
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('models', [])
        except Exception as e:
            print(f"⚠️ Erro ao listar modelos: {e}")
            return []
    
    def model_exists(self, model_name: Optional[str] = None) -> bool:
        """
        Verifica se um modelo específico está disponível.
        
        Args:
            model_name: Nome do modelo (usa self.model se None)
        
        Returns:
            True se o modelo está disponível
        """
        model_name = model_name or self.model
        models = self.list_models()
        
        return any(
            model.get('name', '').startswith(model_name)
            for model in models
        )
    
    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Baixa um modelo do Ollama (se ainda não estiver disponível).
        
        Args:
            model_name: Nome do modelo a baixar (usa self.model se None)
        
        Returns:
            True se o download foi bem-sucedido
        """
        model_name = model_name or self.model
        
        if not self.is_running():
            print("⚠️ Servidor Ollama não está rodando")
            return False
        
        print(f"Baixando modelo '{model_name}' (isso pode levar alguns minutos)...")
        
        try:
            response = requests.post(
                f"{self.api_base}/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutos para download
            )
            
            response.raise_for_status()
            
            # Processa stream de status
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get('status', '')
                    if status:
                        print(f"  {status}")
            
            print(f"✓ Modelo '{model_name}' baixado com sucesso")
            return True
            
        except Exception as e:
            print(f"⚠️ Erro ao baixar modelo: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """
        Gera texto usando o modelo Ollama.
        
        Args:
            prompt: Prompt de entrada
            system: Mensagem de sistema (instruções para o modelo)
            temperature: Temperatura override (usa self.temperature se None)
            max_tokens: Número máximo de tokens a gerar
            stream: Se True, retorna stream de tokens
        
        Returns:
            Dicionário com resposta:
            - 'response': Texto gerado
            - 'model': Modelo usado
            - 'done': Se a geração foi completada
            - 'error': Mensagem de erro (se houver)
        """
        if not self.is_running():
            return {
                'error': 'Servidor Ollama não está rodando',
                'response': None,
                'done': False
            }
        
        # Prepara payload
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': stream,
            'options': {
                'temperature': temperature if temperature is not None else self.temperature
            }
        }
        
        if system:
            payload['system'] = system
        
        if max_tokens:
            payload['options']['num_predict'] = max_tokens
        
        try:
            response = requests.post(
                f"{self.api_base}/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            if stream:
                # Para streaming, retorna o objeto response
                return {'stream': response, 'done': False}
            else:
                # Para não-streaming, retorna o resultado completo
                data = response.json()
                return {
                    'response': data.get('response', ''),
                    'model': data.get('model', self.model),
                    'done': data.get('done', False),
                    'context': data.get('context', [])
                }
        
        except requests.exceptions.Timeout:
            return {
                'error': 'Timeout ao gerar resposta',
                'response': None,
                'done': False
            }
        except Exception as e:
            return {
                'error': f'Erro ao gerar resposta: {str(e)}',
                'response': None,
                'done': False
            }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict:
        """
        Realiza chat usando o formato de mensagens.
        
        Args:
            messages: Lista de mensagens no formato [{'role': 'user/assistant/system', 'content': '...'}]
            temperature: Temperatura override
            stream: Se True, retorna stream
        
        Returns:
            Dicionário com resposta similar ao generate()
        """
        if not self.is_running():
            return {
                'error': 'Servidor Ollama não está rodando',
                'response': None,
                'done': False
            }
        
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': stream,
            'options': {
                'temperature': temperature if temperature is not None else self.temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            if stream:
                return {'stream': response, 'done': False}
            else:
                data = response.json()
                message = data.get('message', {})
                return {
                    'response': message.get('content', ''),
                    'role': message.get('role', 'assistant'),
                    'model': data.get('model', self.model),
                    'done': data.get('done', False)
                }
        
        except Exception as e:
            return {
                'error': f'Erro no chat: {str(e)}',
                'response': None,
                'done': False
            }
    
    def get_health_check(self) -> Dict:
        """
        Retorna status detalhado do Ollama.
        
        Returns:
            Dicionário com informações de saúde:
            - 'running': Se o servidor está rodando
            - 'model_available': Se o modelo configurado está disponível
            - 'models_count': Número de modelos instalados
        """
        health = {
            'running': self.is_running(),
            'host': self.host,
            'model': self.model,
            'model_available': False,
            'models_count': 0
        }
        
        if health['running']:
            health['model_available'] = self.model_exists()
            health['models_count'] = len(self.list_models())
        
        return health


if __name__ == "__main__":
    # Exemplo de uso
    print("=== Chess Oracle - Ollama LLM Engine ===\n")
    
    # Inicializa cliente
    ollama = OllamaLLM(
        host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
        model=os.getenv('OLLAMA_MODEL', 'llama3')
    )
    
    # Verifica saúde
    print("Verificando status do Ollama...\n")
    health = ollama.get_health_check()
    
    print(f"Servidor: {health['host']}")
    print(f"Rodando: {'✓' if health['running'] else '✗'}")
    
    if health['running']:
        print(f"Modelo configurado: {health['model']}")
        print(f"Modelo disponível: {'✓' if health['model_available'] else '✗'}")
        print(f"Total de modelos instalados: {health['models_count']}\n")
        
        # Lista modelos
        if health['models_count'] > 0:
            print("Modelos disponíveis:")
            for model in ollama.list_models():
                print(f"  - {model.get('name', 'unknown')}")
            print()
        
        # Testa geração se o modelo estiver disponível
        if health['model_available']:
            print("Testando geração de texto...\n")
            
            result = ollama.generate(
                prompt="Explain the Sicilian Defense in chess in one sentence.",
                system="You are a chess expert. Be concise.",
                temperature=0.7
            )
            
            if 'error' not in result:
                print(f"Resposta: {result['response']}\n")
            else:
                print(f"Erro: {result['error']}\n")
        else:
            print(f"\n⚠️ Modelo '{health['model']}' não está disponível.")
            print(f"   Baixe o modelo com: ollama pull {health['model']}\n")
    else:
        print("\n⚠️ Servidor Ollama não está rodando.")
        print("   Inicie o servidor com: ollama serve\n")
