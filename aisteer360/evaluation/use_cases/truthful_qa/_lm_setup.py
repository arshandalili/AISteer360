from transformers import PreTrainedModel, PreTrainedTokenizerBase


# Q:/A: chat-template fallback used by the source repo when the model lacks one.
DEFAULT_CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- set default_system = '' -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{{- messages[0]['content'] -}}"
    "{%- set idx = 1 -%}"
    "{%- else -%}"
    "{{- default_system -}}"
    "{%- set idx = 0 -%}"
    "{%- endif -%}"
    "{%- for message in messages[idx:] -%}"
    "{%- if message['role'] == 'user' -%}"
    "{{ '\\n' }}Q: {{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{ '\\n' }}A: {{ message['content'] }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '\\n' }}A:"
    "{%- else -%}"
    "{{ eos_token }}"
    "{%- endif -%}"
)


def configure_for_source_repo(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = True
    model.eval()
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
