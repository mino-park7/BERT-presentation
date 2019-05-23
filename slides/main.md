name: inverse
class: center, middle, inverse
layout: true

---
class: titlepage, no-number

# BERT: Pre-trainig of Deep Bidirectional Transformers for Language Understanding

## .author[Minho Park]

### .small[.white[] <br/> ]

### .x-small[https://mino-park7.github.io/BERT-presentation/]

---
layout: false

## BERT

.footnote[(Slide credit: [BERT paper](https://arxiv.org/abs/1810.04805))]

- 최근에 NLP 연구분야에서 핫한 모델인 BERT 논문을 읽고 정리하는 포스트입니다.
- 구성은 논문을 쭉 읽어나가며 정리한 포스트기 때문에 논문과 같은 순서로 정리하였습니다.
- Tmax Data AI 연구소에서 제가 진행한 세미나 동영상도 첨부합니다.

--

<iframe width="560" height="315" src="https://www.youtube.com/embed/2b7_iq8rAVY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## Abstract

- BERT : Bidirectional Encoder Representations form Transformer
  - 논문의 제목에서 볼 수 있듯이, 본 논문은 .red[*"Attention is all you need(Vaswani et al., 2017)"*([arxiv](https://arxiv.org/abs/1706.03762))]에서 소개한 **Transformer** 구조를 활용한 **Language Representation**에 관한 논문입니다.
  - **Transformer**에 대한 자세한 구조를 알고 싶은 분은 위 논문을 읽어보시거나, 다음 블로그 [MChromiak's blog](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XBDvwBMzbOT)를 참고하시면 좋을 듯 합니다.
- BERT는 기본적으로, wiki나 book data와 같은 대용랑 **unlabeled data**로 모델을 미리 학습 시킨 후, 특정 *task*를 가지고 있는 *labeled data*로 **transfer learning**을 하는 모델입니다.
  
---

## Abstract

- BERT이전에 이러한 접근 방법을 가지는 모델이 몇가지 있었습니다. - .red[대용량 unlabeld corpus]를 통해 **.red[language model]**을 학습하고, 이를 토대로 뒤쪽에 특정 *task*를 처리하는 network를 붙이는 방식(ELMo, OpenAI GPT...)
- 하지만 BERT 논문에서는 이전의 모델의 접근 방식은 .red[**shallow bidirectional**] 또는 .red[**unidirectional**]하므로 .red[language representation이 부족]하다고 표현하였습니다.
- 게다가 BERT는 특정 task를 처리하기 위해 새로운 network를 붙일 필요 없이, BERT 모델 자체의 .red[**fine-tuning**]을 통해 해당 task의 *state-of-the-art*를 달성했다고합니다.

---

## 1. Introduction

- Introduction에서는 BERT와 비슷한 접근 방식을 가지고 있는 기존 model에 대한 개략적인 소개를 합니다.
- Language model pre-training은 여러 NLP task의 성능을 향상시키는데에 탁월한 효과가 있다고 알려져 있습니다. (Dai and Le, 2015; Peters et al., 2018, 2018; Radford et al., 2018; ...)
- 이러한 NLP task는 token-level task인 Named Entity Recognition(NER)에서부터 [SQuAD question answering task](https://arxiv.org/abs/1606.05250)와 같은 task까지 광범위한 부분을 커버합니다

---

## 1. Introduction

- 이런 **pre-trained language representation**을 적용하는 방식은 크게 두가지 방식이 있습니다. 하나는 **feature-based** 또 다른 하나는 **fine-tuning** 방식입니다.
  - **feature-based approach** : 특정 task를 수행하는 network에 pre-trained language representation을 추가적인 feature로 제공. 즉, 두 개의 network를 붙여서 사용한다고 보면 됩니다. 대표적인 모델 : ELMo([Peters et al., 2018](https://arxiv.org/abs/1802.05365))
  - **fine-tuning approach** : task-specific한 parameter를 최대한 줄이고, pre-trained된 parameter들을 downstream task 학습을 통해 조금만 바꿔주는(fine-tuning) 방식. 대표적인 모델 : Generative Pre-trained Transformer(OpenAI GPT) ([Radford et al., 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf))
- 앞에 소개한 ELMo, OpenAI GPT는 pre-training시에 동일한 **objective funtion**으로 학습을 수행합니다, 하지만 **BERT**는 새로운 방식으로 **pre-trained Language Representation**을 학습했고 이것은 매우 효과적이었습니다.

---

### 1.1 BERT의 pre-training 방법론

![그림1. BERT, GPT, ELMo (출처 : BERT 논문)](https://mino-park7.github.io/images/2018/12/그림1-bert-openai-gpt-elmo-출처-bert논문.png)
.x-small[그림1. BERT, GPT, ELMo (출처 : BERT 논문)]

- 기존 방법론  

--

  - 앞에 소개한 ELMo, OpenAI GPT는 일반적인 language model을 사용하였습니다. 일반적인 language model이란, 앞의 n 개의 단어를 가지고 뒤의 단어를 예측하는 모델을 세우는 것입니다(n-gram). 

--

  - 하지만 이는 필연적으로 **unidirectional**할 수 밖에 없습니다.

--

  - 이러한 단점을 극복하기 위해 **ELMo**에서는 **Bi-LSTM**으로 양방향성을 가지려고 노력하지만, 굉장히 **shallow**한 양방향성 (단방향 concat 단방향)만을 가질 수 밖에 없었습니다(*그림1*).

---

### 1.1 BERT의 pre-training 방법론

![그림1. BERT, GPT, ELMo (출처 : BERT 논문)](https://mino-park7.github.io/images/2018/12/그림1-bert-openai-gpt-elmo-출처-bert논문.png)
.x-small[그림1. BERT, GPT, ELMo (출처 : BERT 논문)]
- **BERT** pre-training의 새로운 방법론은 크게 2가지로 나눌 수 있습니다. 하나는 **Masked Language Model(MLM)**, 또 다른 하나는 **next sentence prediction**이다.
  
---

### 1.1 BERT의 pre-training 방법론

![그림1. BERT, GPT, ELMo (출처 : BERT 논문)](https://mino-park7.github.io/images/2018/12/그림1-bert-openai-gpt-elmo-출처-bert논문.png)
.x-small[그림1. BERT, GPT, ELMo (출처 : BERT 논문)]
- **Masked Language Model(MLM)** 
  
--

  - MLM은 input에서 무작위하게 몇개의 token을 mask 시킵니다. 그리고 이를 **Transformer** 구조에 넣어서 주변 단어의 context만을 보고 mask된 단어를 예측하는 모델입니다. 

--

  - **OpenAI GPT**도 **Transformer** 구조를 사용하지만, 앞의 단어들만 보고 뒷 단어를 예측하는 **Transformer decoder**구조를 사용합니다(*그림1*). 

---

### 1.1 BERT의 pre-training 방법론

![그림1. BERT, GPT, ELMo (출처 : BERT 논문)](https://mino-park7.github.io/images/2018/12/그림1-bert-openai-gpt-elmo-출처-bert논문.png)
.x-small[그림1. BERT, GPT, ELMo (출처 : BERT 논문)]
- **Masked Language Model(MLM)** 
  - 이와 달리 **BERT**에서는 input 전체와 mask된 token을 한번에 **Transformer encoder**에 넣고 원래 token 값을 예측하므로(*그림1*) **deep bidirectional** 하다고 할 수 있습니다.

--

  - BERT의 MLM에 대해서는 뒷장의 [Pre-training Tasks](#33-pre-training-tasks)에서 더 자세히 설명하겠습니다.

---

### 1.1 BERT의 pre-training 방법론

![그림1. BERT, GPT, ELMo (출처 : BERT 논문)](https://mino-park7.github.io/images/2018/12/그림1-bert-openai-gpt-elmo-출처-bert논문.png)
.x-small[그림1. BERT, GPT, ELMo (출처 : BERT 논문)]
- **next sentence prediction** 

--

  - 이것은 간단하게, 두 문장을 pre-training시에 같이 넣어줘서 두 문장이 이어지는 문장인지 아닌지 맞추는 것입니다. 

--

  - pre-training시에는 50:50 비율로 실제로 이어지는 두 문장과 랜덤하게 추출된 두 문장을 넣어줘서 BERT가 맞추게 시킵니다. 

--

  - 이러한 task는 실제 Natural Language Inference와 같은 task를 수행할 때 도움이 됩니다.

---

## 2. Related Work
- ELMo, OpenAI GPT와 같은 모델이 존재하고, 앞에서 충분히 소개하였기 때문에 생략하도록 하겠습니다. 자세한 내용에서는 [BERT 논문](https://arxiv.org/abs/1810.04805)을 참고 바랍니다.

---
name: centertext
class: center, middle, centertext

## 3. BERT

---

# 3. BERT
- BERT의 아키텍처는 **Attention is all you need**에서 소개된 **Transformer**를 사용하지만, pre-training과 fine-tuning시의 아키텍처를 조금 다르게하여 **Transfer Learning**을 용이하게 만드는 것이 핵심입니다.

---

### 3.1 Model Architecture
.left-column-40[![](images/the-annotated-transformer_14_0.png)]
.right-column-60[- **BERT**는 *transformer* 중에서도 **encoder** 부분만을 사용합니다.] 
.right-column-60[- 이에 대한 자세한 내용은 [Vaswani et al (2017)](https://arxiv.org/abs/1706.03762) 또는 [tensor2tensor의 transformer](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)를 참고 바랍니다.]
  
---

### 3.1 Model Architecture

.left-column-40[![](images/BERT_architecture.png)]
.right-column-60[
- **BERT**는 모델의 크기에 따라 *base* 모델과 *large* 모델을 제공합니다.
  - **$BERT_{base}$** : L=12, H=768, A=12, Total Parameters = 110M
  - **$BERT_{large}$** : L=24, H=1024, A=16, Total Parameters = 340M]

---

### 3.1 Model Architecture

.left-column-40[![](images/BERT_architecture1.png)]
.right-column-60[
- **BERT**는 모델의 크기에 따라 *base* 모델과 *large* 모델을 제공합니다.
  - **$BERT_{base}$** : L=12, H=768, A=12, Total Parameters = 110M
  - **$BERT_{large}$** : L=24, H=1024, A=16, Total Parameters = 340M
- L : transformer block의 layer 수]

---
### 3.1 Model Architecture

.left-column-40[![](images/BERT_architecture2.png)]
.right-column-60[
- **BERT**는 모델의 크기에 따라 *base* 모델과 *large* 모델을 제공합니다.
  - **$BERT_{base}$** : L=12, H=768, A=12, Total Parameters = 110M
  - **$BERT_{large}$** : L=24, H=1024, A=16, Total Parameters = 340M
- L : transformer block의 layer 수
- H : hidden size]

---

### 3.1 Model Architecture

.left-column-40[![](images/BERT_architecture3.png)]
.right-column-60[
- **BERT**는 모델의 크기에 따라 *base* 모델과 *large* 모델을 제공합니다.
  - **$BERT_{base}$** : L=12, H=768, A=12, Total Parameters = 110M
  - **$BERT_{large}$** : L=24, H=1024, A=16, Total Parameters = 340M
- L : transformer block의 layer 수
- H : hidden size
- A : self-attention heads 수]

---

### 3.1 Model Architecture

.left-column-40[![](images/BERT_architecture3.png)]
.right-column-60[
- **BERT**는 모델의 크기에 따라 *base* 모델과 *large* 모델을 제공합니다.
  - **$BERT_{base}$** : L=12, H=768, A=12, Total Parameters = 110M
  - **$BERT_{large}$** : L=24, H=1024, A=16, Total Parameters = 340M
- L : transformer block의 layer 수
- H : hidden size
- A : self-attention heads 수
- feed-forward/filter size = 4H]


---

### 3.1 Model Architecture
- 여기서 **$BERT_{base}$** 모델의 경우, **OpenAI GPT**모델과 *hyper parameter*가 **동일**합니다. 여기서 BERT의 저자가 의도한 바는, 모델의 하이퍼 파라미터가 동일하더라도, **pre-training concept**를 바꾸어 주는 것만으로 훨씬 높은 성능을 낼 수 있다는 것을 보여주고자 하는 것 같습니다.
- **OpenAI GPT**모델의 경우 *그림1*에서 볼 수 있듯, next token 만을 맞추어내는 기본적인 *language model* 방식을 사용하였고, 그를 위해 **transformer decoder**를 사용했습니다. 하지만 **BERT**는 **MLM**과 **NSP**를 위해 **self-attention**을 수행하는 **transformer encoder**구조를 사용했음을 알 수 있습니다.
- 실제로 대부분의 **NLP task SOTA**는 **BERT_large**모델로 이루어 냈습니다.

---

## The Bad

- Syntactic parsing
  - No included grammer (not a black box)
- Feature/dependency parsing
  - No included feature grammer
- The sem package
  - Toy only (lambda-calculus & first order logic)
- Lots of extra stuff
  - Papers, chat programs, alignments, etc

.footnote[(Slide credit: [Bengfort's slide](https://www.slideshare.net/BenjaminBengfort/natural-language-processing-with-nltk?next_slideshow=2))]

---

name: centertext
class: center, middle, centertext

## Let's try NLTK!

---

## Installation

```python
$ pip install -U nltk
$ python
>>> import nltk
>>> nltk.download()
NLTK Downloader
---------------------------------------------------------------------------
    d) Download   l) List    u) Update   c) Config   h) Help   q) Quit
---------------------------------------------------------------------------
Downloader> d
Download which package (l=list; x=cancel)?
  Identifier>

```

---

## Corpus - `nltk.corpus` module

- Corpus
  - Large collection of text
  - Raw or categorized
  - Concentrate on a topic or open domain
- Examples
  - Brown - first, largest corpus, categorized by genre
  - Webtext - reviews, forums, etc
  - Reuters - news corpus
  - Inaugural - US presidents' inaugural addresses
  - udhr - multilingual
- Available corpora can be seen at [http://www.nltk.org/nltk_data/](http://www.nltk.org/nltk_data/)

.footnote[(Slide credit: [Cioroianu's NLTK tutorial](http://www.nyu.edu/projects/politicsdatalab/localdata/workshops/NLTK_Presentation.pdf))]

---

## Corpus - `nltk.corpus` module

For example, to read a list of the words in the Brown Corpus, use `nltk.corpus.brown.words()`

```python
>>> from nltk.corpus import brown
>>> print(", ".join(brown.words()))
The, Fulton, County, Grand, Jury, said, ...
```

--

or in sentences `sents()`

```python
>>> brown.sents()
[[u'The', u'Fulton', ...], [u'The', u'jury', ...], ...]
```

--

with tagging `tagged_words()`

```python
>>> brown.tagged_words()
[(u'The', u'AT'), (u'Fulton', u'NP-TL'), ...]
```

---

## Tokenization - `nltk.tokenize` module

A sentence can be split into words using `word_tokenize()`

```python
>>> from nltk.tokenize import word_tokenize, sent_tokenize
>>> sentence = "All work and no play makes jack a dull boy, all work and no play"
*>>> tokens = word_tokenize(sentence)
>>> tokens
['All', 'work', 'and', 'no', 'play', 'makes', 'jack', 'a',
'dull', 'boy', ',', 'all', 'work', 'no', 'play']
```

--

Same principle can be applied to sentences via `sent_tokenize()`
```python
>>> sentence = "All work and no play makes jack a dull boy. All work and no play"
*>>> tokens = sent_tokenize(sentence)
>>> tokens
['All work and no play makes jack dull boy.', 'All work and no play']
```

--

`word_tokenize()` and `sent_tokenize()` are NLTK's recommended tokenizer (`TreebankWordTokenizer` + `PunkSentenceTokenizer`)

---

## Tokenization - `nltk.tokenize` module

Also provide Twitter-aware tokenizer `TweetTokenizer()`
```python
>>> from nltk.tokenize import TweetTokenizer
>>> tknzr = TweetTokenizer()
>>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
>>> tknzr.tokenize(s0)
['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']
```

--

And [Mose Tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) `MosesTokenizer()`

```pythonn
>>> from nltk.tokenize.moses import MosesTokenizer
>>> tokenizer = MosesTokenizer()
>>> text = u'This, is a sentence with weird» symbols… appearing everywhere¿'
>>> tokenizer.tokenize(text)
[u'This', u',', u'is', u'a', u'sentence', u'with', u'weird', u'»', u'symbols', u'…', u'appearing', u'everywhere', u'¿']
```

---

## Stemming and lemmatization - `nltk.stem` module

.center.img-33[![](images/word-stem.png)]

--

Words can be stemmed via `Stemmer()`

```python
>>> from nltk.stem import PorterStemmer
>>> words = ["game", "gaming", "gamed", "games"]
*>>> ps = PorterStemmer()
>>> [ps.stem(word) for word in words]
['game', 'game', 'game', 'game']
```

---

## Stemming and lemmatization - `nltk.stem` module

</br>

.center.img-70[![](images/word-lemm.png)]

</br>

--

We can also lemmatize via `WordNetLemmatizer()`

```python
>>> from nltk.stem import WordNetLemmatizer
>>> words = ["game", "gaming", "gamed", "games"]
*>>> wnl = WordNetLemmatizer()
>>> [wnl.lemmatize(word) for word in words]
['game', 'gaming', 'gamed', u'game']
```

---

## Tagging - `nltk.tag` module

<!--
.center.img-50[![](images/pos-tagging.jpg)]
-->

A sentence can be tagged using `Tagger()`

```python
>>> nltk.corpus import brown
>>> from nltk.tag import UnigramTagger
*>>> tagger = UnigramTagger()
>>> sent = ['Mitchell', 'decried', 'the', 'high', 'rate', 'of', 'unnemployment']
*>>> tagger.tag(sent)
[('Mitchell', u'NP'), ('decried', None), ('the', u'AT'), ('high', u'JJ'), ('rate', u'NN'), ('of', u'IN'), ('unemployment', None)]
```

--

Or simply use NLTK's recommended tagger via `pos_tag()`

```python
>>> from nltk.tag import pos_tag
>>> from nltk.tokenize import word_tokenize
*>>> pos_tag(word_tokenize("John's big idea isn't all that bad."))
[('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'),
("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]
```

---

## Parsing - `nltk.parser` module

</br>

.center.img-70[![](images/syntactic-parsing.png)]

---

## Parsing - `nltk.parser` module

Provide wrapper for CoreNLP parser `CoreNLPParser()`

```python
from nltk.parse.corenlp import CoreNLPParser
>>> parser = CoreNLPParser(url='http://localhost:9000')
>>> next(
...     parser.raw_parse('The quick brown fox jumps over the lazy dog.')
... ).pretty_print()
                     ROOT
                      |
                      S
       _______________|__________________________
      |                         VP               |
      |                _________|___             |
      |               |             PP           |
      |               |     ________|___         |
      NP              |    |            NP       |
  ____|__________     |    |     _______|____    |
 DT   JJ    JJ   NN  VBZ   IN   DT      JJ   NN  .
 |    |     |    |    |    |    |       |    |   |
The quick brown fox jumps over the     lazy dog  .
```

---

## Parsing - `nltk.parser` module

</br>
</br>
</br>
</br>

.center.img-90[![](images/dependency-parsing.png)]

.footnote[(Image credit: [Nathan Schneider](http://people.cs.georgetown.edu/nschneid/))]

---

## Parsing - `nltk.parser` module

Provide wrapper for CoreNLP dependency parser `CoreNLPDependencyParser()`

```python
from nltk.parse.corenlp import CoreNLPDependencyParser
>>> parser = CoreNLPDependencyParser(url='http://localhost:9000')
>>> parse, = dep_parser.raw_parse(
...     'The quick brown fox jumps over the lazy dog.'
... )
>>> print(parse.to_conll(4))
The     DT      4       det
quick   JJ      4       amod
brown   JJ      4       amod
fox     NN      5       nsubj
jumps   VBZ     0       ROOT
over    IN      9       case
the     DT      9       det
lazy    JJ      9       amod
dog     NN      5       nmod
.       .       5       punct
```

---

## Other libraries?

- [spaCy](https://spacy.io/)
- [unidecode](https://github.com/iki/unidecode)
- [pyEnchant](http://pythonhosted.org/pyenchant/)
- [gensim](https://radimrehurek.com/gensim/)
- [fastText](https://github.com/facebookresearch/fastText)

---

## spaCy: Industrial-Strength NLP in Python

.center.img-66[![](images/spacy-main.png)]

- Minimal and optimized!
  - One algorithm (the best one) for each purpose
- Lightning-fast (written in Cython)

.footnote[(Image credit: [spaCy](https://spacy.io/))]

---

## Detailed Speed Comparision

Per-document processing time of various spaCy functionalities against other NLP libraries

.center.img-77[![](images/spacy-benchmark.png)]

.footnote[(Image credit: [spaCy](https://spacy.io/))]

---

## Parse Accuracy

Google's [SyntaxNet](https://github.com/tensorflow/models/tree/master/syntaxnet) is the winner

.center.img-77[![](images/spacy-parse-accuracy.png)]

.footnote[(Image credit: [spaCy](https://spacy.io/))]

---

## Named Entity Comparison

[Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) is the winner

.center.img-77[![](images/spacy-ner-accuracy.png)]

.footnote[(Image credit: [spaCy](https://spacy.io/))]

---

## unidecode

ASCII transliterations of Unicode text

```python
>>> from unidecode import unnidecode
>>> unidecode(u'파이콘')
'paikon'
```

--

## pyEnchant

Spellchecking library for python

```python
>>> import enchant
>>> en_dict = enchant.Dict("en_US")
>>> en_dict.check("apple")
True
```

---

## gensim

Python library for topic modelling

```python
>>> from gensim.models.ldamulticore import LdaMulticore
>>> lda = LdaMulticore(corpus, id2word=id2word, num_topics=num_topics, ...)
using serial LDA version on this node
running online LDA training, 100 topics, 1 passes over the supplied corpus of 3931787 documents, updating model once every 10000 documents
...
>>> lda.print_topics(20)
topic #0: 0.009*river + 0.008*lake + 0.006*island + 0.005*mountain + 0.004*area + 0.004*park + 0.004*antarctic + 0.004*south + 0.004*mountains + 0.004*dam
topic #1: 0.026*relay + 0.026*athletics + 0.025*metres + 0.023*freestyle + 0.022*hurdles + 0.020*ret + 0.017*divisão + 0.017*athletes + 0.016*bundesliga + 0.014*medals
...
```

--

## fastText

Library for fast text representation and classification

```bash
$ ./fasttext skipgram -input data.txt -output model
$ ./fasttext print-word-vectors model.bin < queries.txt
```
---

name: last-page
class: center, middle, no-number
## Thank You!
#### [@bckim92][bckim92-gh]

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]

[bckim92-gh]: https://github.com/bckim92


