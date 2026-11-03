The hardest moment for me. What was it? The 27-overlapping-texts failure that surfaced hours before shipping.

The biggest surprise for me. The val/test Procedure correction. Also, the LLM failing specifically on domain-named classes

One habit for the next project. Write the leakage test before the model because I still struggle about the leakage notion.

What transfers. 
To NOTAM summarization v2 => DistilBERT is an encoder-only model designed for understanding language rather than generating new text, which means it cannot perform abstractive summarization on its own. However, you can use DistilBERT for extractive summarization (selecting and ranking key sentences from a source text) or combine it in hybrid architectures. Moving from a classifier (V1) to a full summarization system (V2) means transitioning from predicting labels to generating text. However, you do not need to start from scratch. You can reuse your infrastructure, data pipelines, domain-specific text preprocessing, and the classifier itself as a quality filter or extractive baseline.

To an aviation AI job => Thanks to this project, I can define what is NOTAM. Also, I have discovered a real-life problem in the field and see the opportunities of using AI/ML as solution to help Airmen and the industry mainly. I can talk about the NOTAM taxonomy and its four operational domains (runways/navaids/airspace/wildlife) and about why class-conditional vocabulary is enough for 10 of 13 classes but not the tail