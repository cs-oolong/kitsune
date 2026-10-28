# Design: dimensionamento automático e CLI enxuta

Status: implementado (2026-10-27). O solucionador vive em `src/kitsune/estimate.py` (`solve`).

## 1. Problema

Hoje `kitsune start` tem nove flags e `kitsune suggest` recomenda um comando com
`--gpu-util 0.3 --max-len 4096`, dois valores arbitrários que o usuário quase
nunca quer escolher na mão:

- `--max-len 4096` corta o contexto do modelo sem motivo (o nativo de um
  Qwen2.5 é 32k).
- `--gpu-util 0.3` é um chute conservador: às vezes desperdiça GPU, às vezes
  não cabe e obriga o usuário a tentar valores na mão.
- `--partition` e `--gpus` também são escolhas que o `suggest` já sabe fazer
  melhor que o usuário.

## 2. Ideia central

Um único **solucionador** em `estimate.py` que, dado o modelo (Hugging Face) e
o inventário do cluster (Slurm), devolve um **plano completo**:

```
Plan = (partition, gpu_type, gpus, gpu_util, max_len, dtype, diagnostics)
```

- `suggest` passa a só explicar o plano (dry-run).
- `start` passa a calcular o plano e enviar o job, sem pedir nada além do
  modelo. As flags atuais viram **overrides opcionais**, não o caminho normal.

`job.py` não muda: continua recebendo valores concretos já resolvidos.

## 3. Contexto (max-len)

O teto real de contexto é o nativo do modelo, lido do `config.json` que
`fetch_model_info` já baixa hoje:

- `ModelInfo` ganha `native_max_len` (`max_position_embeddings`; se ausente,
  cai para um piso com aviso).
- O padrão de `--max-len` vira `None` = "o máximo que o modelo e a GPU
  permitirem".

O vLLM (engine V1 do 0.8.5) não pré-aloca KV por requisição; mas recusa subir
se o pool de KV não couber **uma** sequência de `max_model_len`. É exatamente o
que o estimador já modela. Para cada candidato (partição, tp):

```
ctx_fit(tp) = ( tp × (mem_gib × TETO − OVERHEAD_GIB) × GiB − pesos_bytes )
              / kv_bytes_per_token
ctx_escolhido = min(native_max_len, ctx_fit)
```

Nota: o overhead por GPU (ativações do profiling, CUDA graphs, buffers do vLLM)
é uma constante em `estimate.py` (`OVERHEAD_GIB = 8.0`), medida no cluster com o
vLLM 0.11 — na era do 0.8.5, 2 GiB bastavam.

- Se `ctx_escolhido == native`: plano serve o contexto inteiro do modelo.
- Se não: o plano é aceito com **aviso explícito** ("contexto limitado a X;
  o nativo Y não cabe no teto de gpu-util Z; use --gpu-util ou mais GPUs").
- Se nem o piso (`MIN_CONTEXT = 4096`) couber em lugar nenhum: recusa com as
  razões, como hoje.

## 4. gpu-util

Inverte a direção do cálculo: em vez de escolher util e testar o encaixe,
calcula o encaixe e deriva o util mínimo:

```
precisa_gib(tp, ctx) = (pesos + kv × ctx) / tp / GiB + OVERHEAD_GIB  # 8 GiB medidos
util(tp, ctx)        = precisa_gib / mem_gib        # arredondado para cima, 2 casas
```

- Viável se `util ≤ GPU_UTIL_CEILING` (constante nova em `config.py`, **0.9** —
  o próprio padrão do vLLM).
- O teto 0.9 pressupõe GPU exclusiva via gres do Slurm. Se o RECOD
  compartilhar GPUs entre jobs, baixamos a constante (ver "Questões em aberto").
- `--gpu-util` continua existindo como override (pula a derivação).

## 5. CLI enxuta

Fluxo diário fica:

```bash
kitsune setup --model autor/repo
kitsune start --model autor/repo    # resolve tudo, mostra o plano, envia
kitsune chat "..."
```

| Flag do `start` | Hoje | Depois |
|---|---|---|
| `--model` | default 0.5B | igual |
| `--partition` | default `l40s` | **auto** (override opcional) |
| `--gpus` | default 1 | **auto** (override opcional) |
| `--gpu-util` | default 0.3 | **auto** (override opcional) |
| `--max-len` | default 4096 | **auto** = nativo do modelo (override opcional) |
| `--time` | default 1h | igual (decisão do usuário) |
| `--trust-remote-code` | flag | igual |
| `--cpus`, `--mem` | flags | **saem da CLI**; viram constantes em `config.py` |

`suggest` fica só com `--model` no uso normal; `--max-len`/`--gpu-util`/
`--gpus`/`--partition` continuam nele como modo "e se?".

Mudança de comportamento a registrar: `start` passa a consultar o Hugging Face
e o cluster antes de enviar (como `suggest` já faz). Sem rede, falha com
mensagem clara sugerindo passar as flags explícitas. `config.py` perde
`DEFAULT_PARTITION`, `DEFAULT_GPUS`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`
e ganha `GPU_UTIL_CEILING`, `MIN_CONTEXT`.

### Ranking dos candidatos

Para cada partição utilizável e cada tp válido (divide as cabeças de atenção):
ordena por (1) serve o contexto nativo, (2) tem `tp` GPUs livres agora num nó,
(3) menos GPUs, (4) menor util derivado, (5) nome da partição. `start` imprime
o plano escolhido (uma linha) e as alternativas, como `suggest` faz hoje.

## 6. Concorrência (experimento multi-chat)

O vLLM é exatamente a ferramenta certa para isso, sem mudança nenhuma no
kitsune:

- **Continuous batching**: requisições paralelas ao mesmo servidor são
  agrupadas a cada passo de decode; a vazão sobe com a concorrência até o
  limite de compute/memória da GPU.
- **Prefix caching** (ligado por padrão na V1): chats que dividem o mesmo
  system prompt reutilizam os blocos de KV dele. Um experimento multi-chat com
  um `system_prompt.txt` comum ganha isso de graça.
- O pool de KV é **compartilhado** entre as sequências concorrentes, com páginas
  alocadas sob demanda. O solucionador reserva KV para **uma** sequência de
  tamanho máximo; chats reais ficam muito abaixo disso, então um pool assim
  atende vários chats paralelos. Se encher, a V1 preempta e recomputa
  (latência sobe, nada falha). Para muitos chats longos e paralelos: suba
  `--gpus`/`--gpu-util` ou baixe `--max-len`.

Caminho de uso: `kitsune tunnel` + clientes próprios batendo em
`http://127.0.0.1:8000/v1` com concorrência. O `kitsune chat` continua
sequencial por invocação.

## 7. Plano por arquivo

- `estimate.py`: `ModelInfo.native_max_len`; novo `solve(...) -> (Plan, alternativas, rejeições)`;
  `suggest` atual vira wrapper de apresentação sobre `solve`.
- `cli.py`: `cmd_start` chama `solve` quando as flags não são passadas
  (defaults `None` = auto); imprime o plano antes de enviar. Flags `--cpus`/
  `--mem` removidas.
- `config.py`: troca de constantes (acima); `GPU_SPECS` e restantes intactos.
- `job.py`: sem mudança (recebe valores concretos).
- `tests/`: testes do solucionador com `ModelInfo`/`Partition` sintéticos
  (nativo cabe / clamp / recusa; util derivado; overrides vencem o auto);
  ajustar os testes de `suggest`/`start` existentes.
- `README.md`: reescrever a seção Uso com o fluxo novo.

## 8. Fora de escopo

- `max_tokens` por resposta (2048 em `chatlog/turn.py`) não muda. Um follow-up
  possível é omitir o campo quando o usuário não quiser teto de geração.
- Template do job, túnel, ledger de chats: intactos.

## 9. Questões em aberto

- **GPUs do RECOD são exclusivas ou compartilhadas?** O teto 0.9 assume
  exclusivo. Verificar com um job de sonda (`nvidia-smi` durante outro job) e
  ajustar `GPU_UTIL_CEILING`.
- **RoPE scaling/YaRN**: `max_position_embeddings` pode ser menor que o
  contexto efetivo estendido. vLLM deriva o valor correto sozinho; usar o campo
  cru é conservador e seguro para o estimador.
- Futuro: flag `--chats N` para dimensionar o pool de KV para N sequências
  concorrentes de contexto cheio.
- Futuro: gravar o `config.json` do modelo no `setup` para `start` não precisar
  de rede.
