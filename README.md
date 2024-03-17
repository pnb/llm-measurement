# Help-seeking measurement with LLMs

Requires server sub-package of *llama-cpp-python*, which mimics the OpenAI API.

## Package installation

```bash
conda create -n llama python=3.11.4 pandas
conda activate llama
pip install llama-cpp-python[server]==0.2.14
pip install guidance==0.0.64
pip install --upgrade openai==0.28.0  # Actually a downgrade, but needed
```

Models can be downloaded from <https://huggingface.co> (e.g., [LLaMA2-chat](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF), the best-performing model for help-seeking detection in our tests).

## Running it

First, start the LLM server. Options may be changed slightly depending on what model you use, and how large your texts are (`--n_ctx` must be a bit larger than the longest one to hold generated output as well as input).

```bash
python -m llama_cpp.server \
  --model ~/models/llama-2-13b-chat.Q5_K_M.gguf \
  --use_mmap false --use_mlock false --n_threads 4 --n_ctx 1000 --seed 1
```

Then, run the script to apply the LLM to your input using the *Guidance* library to enforce correct output format. Your input CSV file must have a column named "message" in it; all other columns, if present, will be ignored.

```bash
python apply_guidance.py input.csv prompt_helpseeking-llamachat.txt output.csv
```

Or run `python apply_guidance.py -h` to see further usage instructions.

## Citation

Bosch, N., Reyes Denis, T., & Perry, M. (in press). Teacher learning online: Detecting patterns of engagement. *Proceedings of the 18th International Conference of the Learning Sciences - ICLS 2024*.

[Paper PDF](https://pnigel.com/papers/bosch-inpress-VGBNAUH8.pdf)
