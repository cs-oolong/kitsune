# kitsune

Sobe um modelo de linguagem num nó de GPU do cluster RECOD por tempo limitado (1 hora por padrão) e conversa com ele pelo laptop. O modelo responde pela API no formato da OpenAI, então qualquer programa que fale com essa API pode usá-lo.

Tudo passa por `ssh headnode`. O kitsune nunca roda no login node.

## Instalação

Você precisa de:

- [uv](https://docs.astral.sh/uv/) instalado no laptop.
- Uma conta no cluster RECOD, com acesso por SSH ao headnode.
- Um alias `headnode` no `~/.ssh/config`. A multiplexação (`ControlMaster`) faz com que você não precise digitar a senha a cada comando:

```
Host headnode
    HostName <endereço do headnode>
    User <seu usuário>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

No laptop, uma vez (rode de novo para atualizar):

```bash
uv tool install --force git+https://github.com/cs-oolong/kitsune
```

## Uso

```bash
kitsune setup                 # uma vez: uv, vLLM e o modelo no headnode
kitsune models                # modelos baixados no headnode, com tamanho e status
kitsune cluster               # GPUs livres por partição, nós fora do ar e fila
kitsune suggest --model Qwen/Qwen2.5-7B-Instruct   # qual partição e quantas GPUs pedir
kitsune start                 # envia o job ao Slurm (1 hora, 1 GPU na l40s)
kitsune start --partition rtx8000 --gpus 2 --gpu-util 0.3
kitsune status                # fila, carregando, pronto
kitsune chat "oi, tudo bem?"                             # sem -c: chat novo (usa system_prompt.txt)
kitsune chat --system "Responda em português." "..."     # chat novo com instrução de sistema
kitsune chat -c chat-20261027T194108Z-16b5a5 "continua"  # continua um chat existente (precisa existir)
kitsune chats                 # lista os chats em ledger-kitsune/
kitsune history              # mostra o chat kitsune mais recente
kitsune history <ID>          # mostra um chat específico
kitsune tunnel                # abre um túnel para outros programas usarem a API
kitsune stop                  # cancela antes da 1 hora
```

Opções úteis: `--model autor/repo`, `--partition`, `--gpus`, `--mem`, `--time`, `--gpu-util`, `--max-len`. O padrão de modelo é `Qwen/Qwen2.5-0.5B-Instruct`, pequeno, para testar o fluxo. Todos os padrões ficam em `src/kitsune/config.py`.

Para escolher a partição e o número de GPUs, rode `kitsune suggest` antes de `kitsune start`. Ele consulta o Hugging Face (tamanho dos pesos e formato das camadas) e o estado atual do cluster. Calcula a memória por GPU para cada número válido de GPUs (`--tensor-parallel-size` precisa dividir o número de cabeças de atenção) e sugere a partição com menos fila entre as que cabem agora. Ele também descarta partições que o vLLM não suporta: GPUs com compute capability abaixo de 7.5 (mínimo da documentação do vLLM), como a P5000.

`kitsune start` lê a partição no cluster antes de enviar o job. Recusa partição inexistente e número de GPUs acima do que um nó tem. Usa `--dtype half` em GPUs com compute capability abaixo de 8.0, como as Turing (RTX 8000 e RTX 5000), porque o vLLM não roda bf16 nelas.

## Chats

- `kitsune chat` sem `-c` cria um chat novo. A instrução de sistema vem de `--system` ou do arquivo `system_prompt.txt` na pasta atual (ou do caminho em `--system-prompt-file`). Se nenhum dos dois existir, o comando falha.
- Com `-c ID`, o chat precisa existir em `ledger-kitsune/` na pasta de onde você roda o comando. Ele continua com a instrução de sistema que tinha quando foi criado.
- Cada chat é de um modelo só. Continuar um chat com outro modelo é recusado.
- Cada chat é um arquivo `<id>.json` em `ledger-kitsune/`. O ID tem o formato `chat-AAAAMMDDTHHMMSSZ-xxxxxx`. Um chat só é gravado quando a resposta chega, então uma falha não deixa lixo.
- `kitsune history` sem ID mostra o chat kitsune mais recente.

## Usar a API de outros programas

`kitsune tunnel` abre um túnel até o servidor em `http://127.0.0.1:8000/v1` (use `--port` para outra porta) e fica aberto até Ctrl-C. A chave fica em `~/.local/state/kitsune/api_key`, com permissão 600. Em outro terminal:

```bash
export KITSUNE_API_KEY="$(cat ~/.local/state/kitsune/api_key)"
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $KITSUNE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "oi"}]}'
```

Para qualquer programa que aceite uma API no formato da OpenAI, use como URL base `http://127.0.0.1:8000/v1`, a chave acima e como modelo o nome que `kitsune tunnel` mostra em `Modelo:`.

## Configuração

- `KITSUNE_HOST`: alias SSH do headnode. Padrão: `headnode`.
- `HF_TOKEN`: token do Hugging Face, usado por `kitsune suggest` para consultar modelos que exigem login.

## Como funciona

- `setup` instala o vLLM num venv em `~/kitsune/venv` e baixa o modelo para `~/kitsune/models/`.
- `start` envia um script de batch pela entrada padrão do `sbatch`. O job escolhe uma porta livre, gera um token novo e sobe `vllm serve` com `--host 0.0.0.0`. Ele grava nó, porta, token e status em `~/kitsune/state/<jobid>.env` (permissão 600). O log do job fica em `~/kitsune/logs/<jobid>.out`.
- `chat` lê o arquivo de estado, abre um túnel `ssh -L` até o nó e chama `/v1/chat/completions` com o token como `Bearer`.
- Quando o limite de tempo acaba, o Slurm encerra o job e o servidor desliga sozinho.

## Desenvolvimento

```bash
uv run --group dev pytest
```

Código em `src/kitsune/`:

- `cli.py`: comandos e saída no terminal.
- `remote.py`: tudo que roda no headnode por SSH (setup, Slurm, modelos).
- `job.py`: o script de batch que sobe o vLLM no nó.
- `estimate.py` e `cluster.py`: memória necessária e leitura das partições.
- `client.py`: túnel SSH.
- `chatlog/`: o ledger de chats e a chamada à API.
