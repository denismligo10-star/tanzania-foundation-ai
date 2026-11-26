
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 16000

tokenizer = Tokenizer(
    models.BPE(unk_token="<unk>")
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

data_dir = Path("data/processed")

files = list(data_dir.glob("*.txt"))

if not files:
    raise FileNotFoundError(
        "No training text files found in data/processed/"
    )

tokenizer.train(
    files=[str(file) for file in files],
    trainer=trainer
)

output_dir = Path("models/tokenizer")
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer.save(str(output_dir / "tokenizer.json"))

print("Tokenizer trained successfully!")
print("Vocabulary size:", tokenizer.get_vocab_size())
