# AI-Article-Tag-Genertor-t5-small <br>(t5-small-machine-articles-tag-generation)

Machine Learning model to generate Tags for Machine Learning related articles. This model is a fine-tuned version of [t5-small](https://huggingface.co/t5-small) fine-tuned on a refined version of [190k Medium Articles](https://www.kaggle.com/datasets/fabiochiusano/medium-articles) dataset for generating Machine Learning article tags using the article textual content as input. While usually formulated as a multi-label classification problem, this model deals with _tag generation_ as a text2text generation task (inspiration and reference: [fabiochiu/t5-base-tag-generation](https://huggingface.co/fabiochiu/t5-base-tag-generation)).
<br><br>
Finetuning Notebook Reference: [Hugging face summarization notebook](https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb).
##  How to use the model
### Installations

```python
pip install transformers nltk
```
### Code

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")

article_text = """
Paige, AI in pathology and genomics

Fundamentally transforming the diagnosis and treatment of cancer
Paige has raised $25M in total. We talked with Leo Grady, its CEO.
How would you describe Paige in a single tweet?
AI in pathology and genomics will fundamentally transform the diagnosis and treatment of cancer.
How did it all start and why? 
Paige was founded out of Memorial Sloan Kettering to bring technology that was developed there to doctors and patients worldwide. For over a decade, Thomas Fuchs and his colleagues have developed a new, powerful technology for pathology. This technology can improve cancer diagnostics, driving better patient care at lower cost. Paige is building clinical products from this technology and extending the technology to the development of new biomarkers for the biopharma industry.
What have you achieved so far?
TEAM: In the past year and a half, Paige has built a team with members experienced in AI, entrepreneurship, design and commercialization of clinical software.
PRODUCT: We have achieved FDA breakthrough designation for the first product we plan to launch, a testament to the impact our technology will have in this market.
CUSTOMERS: None yet, as we are working on CE and FDA regulatory clearances. We are working with several biopharma companies.
What do you plan to achieve in the next 2 or 3 years?
Commercialization of multiple clinical products for pathologists, as well as the development of novel biomarkers that can help speed up and better inform the diagnosis and treatment selection for patients with cancer.
"""

inputs = tokenizer([article_text], max_length=1024, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10,
                        max_length=128)

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

tags = [ tag.strip() for tag in decoded_output.split(",")] 

print(tags)

# ['Paige', 'AI in pathology and genomics', 'AI in pathology', 'genomics']


```


## Dataset Preparation

Over the 190k article dataset from Kaggle, around 12k of them are Machine Learning based and the tags were pretty high level. 
Generating more specific tags would be of use while developing a system for Technical Blog Platforms. 
ML Articles were filtered out and around 1000 articles were sampled. GPT3 API was used to Tag those and then preprocessing on the generated tags was performed to enusre articles with 4 or 5 tags were selected for the final dataset that came around 940 articles.

#### Notebook : [/ML_Articles_Dataset_Generation_GPT3](https://github.com/nandakishormpai/AI-Article-Tag-Genertor-t5-small/blob/main/ML_Articles_Dataset_Generation_GPT3.ipynb)


## Intended uses & limitations

This model can be used to generate Tags for Machine Learning articles primarily and can be used for other technical articles expecting a lesser accuracy and detail. The results might contain duplicate tags that must be handled in the postprocessing of results.

## Results

It achieves the following results on the evaluation set:
- Loss: 1.8786
- Rouge1: 35.5143
- Rouge2: 18.6656
- Rougel: 32.7292
- Rougelsum: 32.6493
- Gen Len: 17.5745

## Training and evaluation data

The dataset of over 940 articles was split across train : val : test as 80:10:10 ratio samples. 


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10
- mixed_precision_training: Native AMP


### Framework versions

- Transformers 4.26.1
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2



