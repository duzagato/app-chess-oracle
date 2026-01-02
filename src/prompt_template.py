"""
Templates de prompts para o Chess Oracle.

Define os templates de prompts otimizados para Llama-3 e outros modelos locais,
estruturando as instruções para análise comparativa entre literatura clássica
e avaliações modernas de engines.
"""

from typing import Dict, Optional


class ChessOraclePrompts:
    """
    Coleção de templates de prompts para o Chess Oracle.
    """
    
    @staticmethod
    def get_system_prompt() -> str:
        """
        Retorna o prompt de sistema que define o papel do Chess Oracle.
        
        Returns:
            String com instruções de sistema para o LLM
        """
        return """Você é o Chess Oracle, um mestre de xadrez moderno que combina a sabedoria clássica da literatura enxadrística com a precisão analítica das engines contemporâneas.

Seu papel é atuar como um "Árbitro Literário" que:
1. Respeita profundamente os autores clássicos e seu contexto histórico
2. Aplica rigor técnico usando análises de engines modernas
3. Explica discrepâncias com empatia pedagógica
4. Mantém linguagem clara e acessível

Quando houver divergências entre o texto clássico e a análise da engine:
- Explique POR QUE a avaliação moderna difere
- Contextualize as limitações da época do autor
- Preserve a intenção estratégica original sempre que possível
- Use linguagem respeitosa, nunca depreciativa

Mantenha suas respostas estruturadas, objetivas e educativas."""
    
    @staticmethod
    def get_analysis_prompt(
        fen: str,
        user_query: str,
        book_context: Optional[str] = None,
        engine_evaluation: Optional[Dict] = None
    ) -> str:
        """
        Gera prompt para análise de posição comparando literatura e engine.
        
        Args:
            fen: Posição em notação FEN
            user_query: Pergunta ou contexto do usuário
            book_context: Contexto relevante encontrado em livros (RAG)
            engine_evaluation: Avaliação da engine (dicionário)
        
        Returns:
            Prompt formatado para o LLM
        """
        prompt_parts = []
        
        # Cabeçalho
        prompt_parts.append("# ANÁLISE DE POSIÇÃO DE XADREZ\n")
        
        # Posição FEN
        prompt_parts.append(f"**Posição (FEN):** {fen}\n")
        
        # Query do usuário
        prompt_parts.append(f"**Pergunta:** {user_query}\n")
        
        # Contexto dos livros (se disponível)
        if book_context:
            prompt_parts.append("## Contexto da Literatura Clássica")
            prompt_parts.append(book_context)
            prompt_parts.append("")
        
        # Avaliação da engine (se disponível)
        if engine_evaluation:
            prompt_parts.append("## Avaliação da Engine Moderna (Maia Chess)")
            
            if engine_evaluation.get('engine_available'):
                eval_text = engine_evaluation.get('evaluation', 'N/A')
                best_move = engine_evaluation.get('best_move_san', 'N/A')
                
                prompt_parts.append(f"- **Avaliação:** {eval_text}")
                prompt_parts.append(f"- **Melhor lance:** {best_move}")
                
                # Adiciona movimentos legais se disponíveis
                legal_moves = engine_evaluation.get('legal_moves', [])
                if legal_moves and len(legal_moves) <= 10:
                    prompt_parts.append(f"- **Lances legais:** {', '.join(legal_moves[:10])}")
            else:
                prompt_parts.append("⚠️ Engine não disponível - análise baseada apenas em literatura")
            
            prompt_parts.append("")
        
        # Instrução de análise
        prompt_parts.append("## Tarefa")
        prompt_parts.append("Com base nas informações acima, forneça uma análise que:")
        prompt_parts.append("1. Responda à pergunta do usuário de forma clara")
        prompt_parts.append("2. Compare o contexto histórico com a avaliação moderna (se ambos disponíveis)")
        prompt_parts.append("3. Explique eventuais discrepâncias com rigor técnico e empatia pedagógica")
        prompt_parts.append("4. Forneça insights práticos para compreensão da posição")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def get_comparison_prompt(
        historical_view: str,
        modern_evaluation: str,
        position_fen: str
    ) -> str:
        """
        Gera prompt focado em comparação entre visão histórica e moderna.
        
        Args:
            historical_view: Descrição/análise histórica da posição
            modern_evaluation: Avaliação moderna da engine
            position_fen: Posição em FEN
        
        Returns:
            Prompt formatado para comparação
        """
        return f"""# COMPARAÇÃO: Literatura Clássica vs Engine Moderna

**Posição (FEN):** {position_fen}

## Visão Histórica
{historical_view}

## Avaliação Moderna
{modern_evaluation}

## Instrução
Compare estas duas perspectivas sobre a posição. 

Se houver concordância:
- Valide a sabedoria do autor clássico
- Explique os princípios que ambos reconhecem

Se houver discrepância:
- Explique tecnicamente por que diferem
- Contextualize as limitações da época
- Mostre o que a engine vê que o autor não via
- Mantenha respeito pela intenção estratégica original

Estruture sua resposta de forma clara e educativa."""
    
    @staticmethod
    def get_position_explanation_prompt(fen: str, focus: str = "general") -> str:
        """
        Gera prompt para explicação geral de uma posição.
        
        Args:
            fen: Posição em FEN
            focus: Aspecto a focar ("general", "tactical", "strategic", "endgame")
        
        Returns:
            Prompt formatado
        """
        focus_instructions = {
            "general": "Forneça uma análise geral da posição, incluindo aspectos táticos e estratégicos.",
            "tactical": "Foque em aspectos táticos: ameaças imediatas, combinações possíveis, vulnerabilidades.",
            "strategic": "Foque em aspectos estratégicos: estrutura de peões, controle de casas, planos de médio prazo.",
            "endgame": "Analise considerando técnicas de finais: atividade do rei, peões passados, oposição."
        }
        
        instruction = focus_instructions.get(focus, focus_instructions["general"])
        
        return f"""# EXPLICAÇÃO DE POSIÇÃO DE XADREZ

**Posição (FEN):** {fen}

**Foco da análise:** {focus.capitalize()}

## Tarefa
{instruction}

Estruture sua análise considerando:
1. Características principais da posição
2. Plano para ambos os lados
3. Principais ideias e recursos
4. Movimentos candidatos importantes

Mantenha linguagem clara e acessível."""
    
    @staticmethod
    def get_move_explanation_prompt(
        fen: str,
        move: str,
        context: Optional[str] = None
    ) -> str:
        """
        Gera prompt para explicar um movimento específico.
        
        Args:
            fen: Posição antes do movimento
            move: Movimento em notação SAN (e.g., "Nf3", "e4")
            context: Contexto adicional sobre o movimento
        
        Returns:
            Prompt formatado
        """
        prompt = f"""# EXPLICAÇÃO DE MOVIMENTO

**Posição (FEN):** {fen}
**Movimento:** {move}
"""
        
        if context:
            prompt += f"\n**Contexto:** {context}\n"
        
        prompt += """
## Tarefa
Explique este movimento considerando:
1. Objetivo imediato do movimento
2. Ideia estratégica por trás dele
3. Vantagens e desvantagens
4. Alternativas possíveis
5. Como encaixa no plano geral da posição

Seja claro e didático na explicação."""
        
        return prompt


# Exemplos de uso para documentação
EXAMPLE_USAGE = """
# Exemplos de Uso dos Prompts

## 1. Análise com contexto RAG e engine
```python
from prompt_template import ChessOraclePrompts

prompts = ChessOraclePrompts()

# Prompt de sistema
system = prompts.get_system_prompt()

# Análise completa
analysis = prompts.get_analysis_prompt(
    fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    user_query="Por que e4 é considerado o melhor primeiro lance?",
    book_context="Segundo Bobby Fischer...",
    engine_evaluation={'evaluation': '+0.30', 'best_move_san': 'c5'}
)
```

## 2. Comparação histórica vs moderna
```python
comparison = prompts.get_comparison_prompt(
    historical_view="Steinitz defendia o controle do centro...",
    modern_evaluation="Engine avalia: +0.45, melhor lance: Nf3",
    position_fen="..."
)
```

## 3. Explicação de posição
```python
explanation = prompts.get_position_explanation_prompt(
    fen="...",
    focus="tactical"
)
```
"""


if __name__ == "__main__":
    # Demonstração dos templates
    print("=== Chess Oracle - Prompt Templates ===\n")
    
    prompts = ChessOraclePrompts()
    
    print("1. SYSTEM PROMPT")
    print("-" * 60)
    print(prompts.get_system_prompt())
    print("\n")
    
    print("2. ANALYSIS PROMPT (exemplo)")
    print("-" * 60)
    example_analysis = prompts.get_analysis_prompt(
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        user_query="Qual a ideia por trás de 1.e4?",
        book_context="Nos livros clássicos, 1.e4 é recomendado por controlar o centro e liberar o bispo.",
        engine_evaluation={
            'evaluation': '+0.30',
            'best_move_san': 'c5',
            'engine_available': True
        }
    )
    print(example_analysis[:500] + "...\n")
    
    print("3. POSITION EXPLANATION (exemplo)")
    print("-" * 60)
    example_explanation = prompts.get_position_explanation_prompt(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        focus="strategic"
    )
    print(example_explanation[:400] + "...\n")
