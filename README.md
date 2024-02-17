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

## Running it

First, start the LLM server. Options may be changed slightly depending on what model you use, and how large your texts are (`--n_ctx` must be a bit larger than the longest one to hold generated output as well as input).

```bash
python -m llama_cpp.server \
  --model ~/models/llama-2-13b-chat.Q5_K_M.gguf \
  --use_mmap false --use_mlock false --n_threads 4 --n_ctx 1000 --seed 1
```

Then, run the script to apply the LLM to your input using the *Guidance* library to enforce correct output format.

```bash
python apply_guidance.py input.csv prompt_helpseeking.txt output.csv
```

Or run `python apply_guidance.py -h` to see further usage instructions.
