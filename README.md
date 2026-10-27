# kitsune

Mini-OpenRouter para o cluster RECOD: sobe um modelo num nó de GPU por tempo limitado (1 hora por padrão) e conversa com ele pelo laptop.

Tudo passa por `ssh headnode`. O kitsune nunca roda no login node.

## Instalação

No laptop, uma vez:

```bash
uv tool install --force git+https://github.com/cs-oolong/kitsune
```

Pré-requisito: o alias `headnode` no `~/.ssh/config` (com a multiplexação ativa, para não pedir senha a cada comando).

## Uso

```bash
kitsune setup                 # uma vez: uv, vLLM e o modelo no headnode
kitsune models                # modelos baixados no headnode, com tamanho e status
kitsune cluster               # GPUs livres por partição, nós fora do ar e fila
kitsune suggest --model Qwen/Qwen2.5-7B-Instruct   # qual partição e quantas GPUs pedir
kitsune start                 # envia o job ao Slurm (1 hora, 1 GPU na l40s)
kitsune start --partition rtx8000 --gpus 2 --gpu-util 0.3
kitsune status                # fila, carregando, pronto
kitsune chat "oi, tudo bem?"  # continua o chat ativo
kitsune chat --new "outro assunto"                       # chat novo, vira o ativo
kitsune chat --system "Responda em português." "..."     # chat novo com instrução de sistema
kitsune chat -c a3f9 "voltando ao assunto anterior"      # continua o chat a3f9
kitsune chats                 # lista os chats (* = ativo)
kitsune history               # mostra o chat ativo
kitsune history a3f9          # mostra um chat específico
kitsune stop                  # cancela antes da 1 hora
```

Opções úteis: `--model autor/repo`, `--partition`, `--gpus`, `--mem`, `--time`, `--gpu-util`, `--max-len`. O padrão de modelo é `Qwen/Qwen2.5-0.5B-Instruct`, pequeno, para testar o fluxo.

`kitsune suggest` consulta o Hugging Face (tamanho dos pesos e formato das camadas) e o estado atual do cluster. Ele calcula a memória por GPU para cada número válido de GPUs (`--tensor-parallel-size` precisa dividir o número de cabeças de atenção) e sugere a partição com menos fila entre as que cabem agora. Ele também descarta partições que o vLLM não suporta: GPUs com compute capability abaixo de 7.5 (mínimo da documentação do vLLM), como a P5000.

`kitsune start` lê a partição no cluster antes de enviar o job. Recusa partição inexistente e número de GPUs acima do que um nó tem. Usa `--dtype half` em GPUs com compute capability abaixo de 8.0, como as Turing (RTX 8000 e RTX 5000), porque o vLLM não roda bf16 nelas.

## Como funciona

- `setup` instala o vLLM num venv em `~/kitsune/venv` e baixa o modelo para `~/kitsune/models/`.
- `start` envia um script de batch pela entrada padrão do `sbatch`. O job escolhe uma porta livre, gera um token novo e sobe `vllm serve` com `--host 0.0.0.0`. Ele grava nó, porta, token e status em `~/kitsune/state/<jobid>.env` (permissão 600).
- `chat` lê o arquivo de estado, abre um túnel `ssh -L` até o nó e chama `/v1/chat/completions` com o token como `Bearer`.
- Cada chat é um arquivo em `~/.local/state/kitsune/chats/<id>.json`, com ID de 4 caracteres hexadecimais. Um chat só é gravado quando a resposta chega, então uma falha não deixa lixo. O chat ativo fica no arquivo `active` nessa mesma pasta.
- Quando o limite de tempo acaba, o Slurm encerra o job e o servidor desliga sozinho.

## Desenvolvimento

```bash
uv run --group dev pytest
```
