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
kitsune suggest --model Qwen/Qwen2.5-7B-Instruct   # mostra o plano sem enviar
kitsune start                 # sobe o job, acompanha até ficar pronto e já segura o túnel aberto
kitsune start --gpus 2 --max-len 8192              # overrides pontuais
kitsune start --no-tunnel      # sai ao ficar pronto, sem segurar o túnel
kitsune start --no-follow      # só envia e sai (para scripts)
kitsune watch                  # reanexa o acompanhamento a qualquer momento
kitsune status                # fila, carregando, pronto; sem job no ar, o destino do último job
kitsune logs                  # últimas 50 linhas do log do job mais recente
kitsune logs 121897 --lines 0 # log inteiro de um job específico
kitsune chat "oi, tudo bem?"                             # sem -c: chat novo (usa system_prompt.txt)
kitsune chat --system "Responda em português." "..."     # chat novo com instrução de sistema
kitsune chat -c chat-20261027T194108Z-16b5a5 "continua"  # continua um chat existente (precisa existir)
kitsune chats                 # lista os chats em ledger-kitsune/
kitsune history              # mostra o chat kitsune mais recente
kitsune history <ID>          # mostra um chat específico
kitsune tunnel                # abre um túnel para outros programas usarem a API
kitsune stop                  # cancela antes da 1 hora
```

O padrão de modelo é `Qwen/Qwen2.5-0.5B-Instruct`, pequeno, para testar o fluxo. Os padrões ficam em `src/kitsune/config.py`.

`kitsune start` calcula o plano sozinho: consulta o Hugging Face (tamanho dos pesos, formato das camadas e contexto nativo do modelo) e o estado atual do cluster, e deriva partição, número de GPUs, gpu-util e comprimento de contexto. Depois de enviar o job, ele já acompanha o status redesenhando a tela a cada segundo (como um `watch`) e, quando o servidor fica pronto, **já abre o `kitsune tunnel` e o segura até Ctrl-C** — que fecha só o túnel; o job continua no ar até o limite de tempo ou um `kitsune stop`. Se o job morre, sai com código 1 mostrando a causa provável. `--no-tunnel` sai ao ficar pronto (comportamento de script), `--no-follow` sai logo após enviar. Sem terminal (saída redirecionada), imprime o status uma vez e sai — nunca bloqueia segurando túnel. `kitsune watch` reanexa o acompanhamento a qualquer momento. O contexto alvo é o nativo do modelo (`max_position_embeddings`), limitado ao que couber no teto de gpu-util (0.9); se não couber, o plano avisa até onde dá. O gpu-util é o mínimo que comporta pesos + KV cache + overhead — nada de tunar na mão. `--partition`, `--gpus`, `--gpu-util` e `--max-len` existem como overrides; passando os quatro, `start` nem consulta o Hugging Face (útil sem rede). `--time` (padrão 1 hora) e `--trust-remote-code` continuam decisões suas.

`kitsune suggest` mostra o mesmo plano sem enviar nada: o recomendado, as alternativas e por que cada partição foi descartada. Com flags (`--partition`, `--gpus`, `--gpu-util`, `--max-len`) vira modo "e se?". O solucionador descarta partições que o vLLM não suporta (compute capability abaixo de 7.5, como a P5000) e usa `--dtype half` em GPUs abaixo de 8.0, como as Turing, porque o vLLM não roda bf16 nelas. O desenho está em `DESIGN.md`.

## Chats

- `kitsune chat` sem `-c` cria um chat novo. A instrução de sistema vem de `--system` ou do arquivo `system_prompt.txt` na pasta atual (ou do caminho em `--system-prompt-file`). Se nenhum dos dois existir, o comando falha.
- Com `-c ID`, o chat precisa existir em `ledger-kitsune/` na pasta de onde você roda o comando. Ele continua com a instrução de sistema que tinha quando foi criado.
- Cada chat é de um modelo só. Continuar um chat com outro modelo é recusado.
- Cada chat é um arquivo `<id>.json` em `ledger-kitsune/`. O ID tem o formato `chat-AAAAMMDDTHHMMSSZ-xxxxxx`. Um chat só é gravado quando a resposta chega, então uma falha não deixa lixo.
- `kitsune history` sem ID mostra o chat kitsune mais recente.

## Usar a API de outros programas

`kitsune tunnel` abre um túnel até o servidor em `http://127.0.0.1:8000/v1` (use `--port` para outra porta) e fica aberto até Ctrl-C. A chave fica em `~/.local/state/kitsune/api_key`, com permissão 600. Enquanto o túnel está aberto, ele se anuncia em `~/.local/state/kitsune/tunnel.json` e **todo `kitsune chat` o reutiliza** — nada de abrir e fechar um túnel SSH a cada turno (a validação é por token, então um túnel de um servidor antigo nunca é usado por engano; sem túnel aberto, o `chat` abre o dele como sempre). Em outro terminal:

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
- `start` envia um script de batch pela entrada padrão do `sbatch`. O job escolhe uma porta livre, gera um token novo e sobe `vllm serve` com `--host 0.0.0.0`. Ele grava nó, porta, token e status em `~/kitsune/state/<jobid>.env` (permissão 600). O log do job fica em `~/kitsune/logs/<jobid>.out` e se lê do laptop com `kitsune logs`.
- Quando não há job no ar, `status` reconstrói o destino do último job (pelo `sacct` e pelo arquivo de estado): se falhou, mostra o código de saída e a causa provável extraída do log; se foi cancelado ou estourou o tempo, diz isso em vez de parecer que nunca existiu.
- `chat` lê o arquivo de estado e chama `/v1/chat/completions` com o token como `Bearer`. Se houver um `kitsune tunnel` aberto para o mesmo servidor (descoberto em `tunnel.json` e validado por `/v1/models`), usa ele; senão, abre um túnel `ssh -L` só para aquela mensagem.
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
