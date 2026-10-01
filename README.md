# _offlineisbetter_

building cpu-only text encoder models.
stop depending on the cloud.
stop buying expensive gpus.

choose to be different. choose to be 🌳 chronically offline 🌳

## try it yourself

don't take our word for it. try it yourself.

```bash
pip install offlineisbetter
```

run `offlineisbetter` by selecting a model.

```bash
offlineisbetter 0.0.0.0 8585 offline-sentiment-small
```

you can then query the api with a json like so.

```python3
requests.post(f"http://127.0.0.1:8585/inference", json={"text": "my input text here"}")
```

## about us

we believe that you shouldn't give your data to faceless companies, that you deserve to run text models locally, and that you shouldn't need to buy expensive hardware. so we're building _offlineisbetter_.

_offlineisbetter_ models are for _encoding_ tasks: sentiment analysis, text tagging, document retrieval, etc. rather than for _decoding_ tasks like autoregressive generation. we believe that it's wasteful and dangerous to depend on cloud apis for frontier language models to do these simple tasks, and it should be almost mindless to download a model to _use_ it without dealing with runtimes or quantization formats.

## models available

currently, we only have one model available: `offline-sentiment-small`. it's a 230m parameter model for text sentiment analysis, finetuned on [stanfordnlp/ssl2](https://huggingface.co/datasets/stanfordnlp/sst2).

## philosophy

succinctly, the core philosophy of _offlineisbetter_ is that parameter-efficient and low-latency models should be easily accessible to everybody. of course hugging face and `transformers.pipeline` allows you to run sentiment analysis in three lines of python, but for more parameter-efficient models, already quantized and with optimized computation graphs.

## license

the source code in this repository is under the mit license. parameters are finetuned from [liquid ai][https://liquid.ai] pretrained models, and therefore are subject to the lfm open license. see `LICENSE` for more information.
