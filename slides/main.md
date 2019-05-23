name: inverse
class: center, middle, inverse
layout: true

---
class: titlepage, no-number

# NLTK: the natural language toolkit

## .author[Byeongchang Kim]

### .small[.white[Sep 7th, 2017] <br/> ]

### .x-small[https://bckim92.github.io/2017f-nlp-talk]

---
layout: false

## Natural Language Processing

.footnote[(Slide credit: [Cioroianu's NLTK tutorial](http://www.nyu.edu/projects/politicsdatalab/localdata/workshops/NLTK_Presentation.pdf))]

- NLP
  - Broad sense: any kind of computer manipulation of natural language
  - From word frequencies to "understanding" meaning

--

- Applications
  - Text processing
  - Information extraction
  - Document classification and sentiment analysis
  - Document similarity
  - Automatic summarizing
  - Discourse analysis


---

## What is NLTK?

.center.img-66[![](images/nltk-main.png)]
.center.small[S. Bird et al. [NLTK: the natural language toolkt][bird-2006], In *COLING-ACL*, 2006]

- Suite of open source Python libraries and programes for NLP
  - Python: open source programming language
- Developed for educational purposes by Steven Bird, Ewan Klein and Edward Loper
- Very good online documentation

[bird-2006]: http://www.aclweb.org/anthology/P04-3031

---

## Some Numbers

- 3+ classification algorithms
- 9+ Part-of-Speech tagging Algorithms
- Stemming algorithms for 15+ languages
- 5+ word tokenization algorithms
- Sentence tokenizers for 16+ languages
- 60+ included corpora

.footnote[(Slide credit: [Perkins's slide](https://www.slideshare.net/japerk/nltk-the-good-the-bad-and-the-awesome-8556908))]

---

## The Good

- Preprocessing
  - segmentation, tokenization, PoS tagging
- Word level processing
  - WordNet, lemmatization, stemming, n-gram
- Utilities
  - Tree, FreqDist, ConditionalFreqDist
  - Streaming CorpusReader objects
- Classification
  - Maximum Entropy, Naive Bayes, Decision Tree
  - Chunking, Named Entity Recognition
- Parsers Galore!
- Languages Galore!

.footnote[(Slide credit: [Bengfort's slide](https://www.slideshare.net/BenjaminBengfort/natural-language-processing-with-nltk?next_slideshow=2))]

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


