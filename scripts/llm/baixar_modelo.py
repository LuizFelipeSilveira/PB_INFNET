"""Baixa o modelo do Hugging Face para a pasta modelos/ do projeto (só precisa rodar uma vez).

Depois do download, src/llm.py carrega o modelo direto dessa pasta, sem acessar a
internet. A pasta modelos/ fica fora do Git (.gitignore), porque tem alguns GB.

Uso, na raiz do projeto:
    python scripts/llm/baixar_modelo.py                      # modelo padrão (Qwen/Qwen3.5-0.8B)
    python scripts/llm/baixar_modelo.py Qwen/Qwen3.5-2B      # outro modelo
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.llm import MODELO_PADRAO, pasta_local  # noqa: E402


def main() -> None:
    from huggingface_hub import snapshot_download

    repo = sys.argv[1] if len(sys.argv) > 1 else MODELO_PADRAO
    destino = pasta_local(repo)
    print(f"Baixando {repo} para {destino.relative_to(RAIZ)} ...")
    snapshot_download(repo_id=repo, local_dir=destino,
                      # só o necessário para o Transformers (sem cópias em outros formatos)
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja",
                                      "*.model", "*.tiktoken"])
    tamanho = sum(f.stat().st_size for f in destino.rglob("*") if f.is_file()) / 1024**3
    print(f"Concluído: {tamanho:.1f} GB em {destino.relative_to(RAIZ)}")
    print("A API e o script de avaliação já vão usar esta cópia local automaticamente.")


if __name__ == "__main__":
    main()
