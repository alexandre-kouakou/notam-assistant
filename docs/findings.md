Make sure src/__init__.py exists in the subfolder before trying to use a python file function in an experimental notebook. 
Without it, Python won't treat src as a package

On your max_length decision. You gave me percentiles but not a decision. Let me push you to make one, because this is a real design choice and the practice of committing to it is the skill.

Your data: 95th percentile is 375 characters / 53 words. 99th percentile is 880 / 131.

Two considerations you should reason through:

Transformers work in tokens, not words or characters. DistilBERT's tokenizer uses WordPiece, which produces roughly 1.3-1.5 tokens per word for English text and often more for text full of abbreviations like NOTAMs — "U/S" and "RWY" get split into subword pieces the tokenizer wasn't optimized for. So a NOTAM at your 95th percentile (53 words) will likely produce 80-120 tokens after tokenization. The 99th percentile might hit 200+.

DistilBERT's max context is 512 tokens. Common practical choices are 128, 256, or 512. Higher max_length means slower training and more GPU memory; too low means truncating your longest inputs.

Given your numbers, 128 tokens is a reasonable v1 max_length. It comfortably fits 95% of your NOTAMs and only truncates the extreme tail. If you find in error analysis that truncation is hurting the long-NOTAM class, you bump to 256 in v2. Start conservative, measure impact, adjust if needed.