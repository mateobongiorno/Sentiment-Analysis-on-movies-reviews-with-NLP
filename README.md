<p><strong>Sentiment Analysis on Movies Reviews</strong></p>

<p><strong>Project Overview</strong></p>

<p>Sentiment analysis, also known as opinion mining, is the process of determining the sentiment or emotional tone behind a given text. In this project, we aim to perform sentiment analysis on movie reviews using natural language processing (NLP) techniques.</p>

<p>The goal is to develop a model that can accurately classify movie reviews as positive or negative based on the sentiment expressed in the text. By analyzing movie reviews, we can gain insights into audience reactions and sentiments towards different movies, which can be valuable for various purposes such as market research, recommendation systems, and understanding audience preferences.</p>

<p>To accomplish this, we will utilize the power of NLP techniques and machine learning algorithms. The project includes tasks such as data preprocessing, feature extraction, model training, and evaluation. We will leverage popular libraries and tools like NLTK, Scikit-learn, and Spacy to perform various NLP operations and build a sentiment analysis model.</p>

<p>The provided Jupyter notebook, <code>Sentiment_Analysis_NLP.ipynb</code>, contains the project statement, a detailed description of the tasks involved, and the sections of code that need to be completed to solve the project. It also includes some basic tests to ensure the correctness of the implemented functionality.</p>

<p><strong>Install</strong></p>

<p>You can use Docker to easily install all the needed packages and libraries:</p>

<pre><code>$ docker build -t nlp_project -f docker/Dockerfile .
</code></pre>

<p><strong>Run Docker</strong></p>

<pre><code>$ docker run --rm --net host -it
-v $(pwd):/home/app/src
nlp_project
bash
</code></pre>

<p><strong>Run Project</strong></p>

<p>Regardless of whether you are inside or outside a Docker container, to execute the project, you need to launch a Jupyter notebook server by running:</p>

<pre><code>$ jupyter notebook
</code></pre>

<p>Then, open the file <code>Sentiment_Analysis_NLP.ipynb</code>, which contains the project statement, description, and the parts of the code you need to complete.</p>

<p><strong>Tests</strong></p>

<p>We have provided some basic tests in <code>Sentiment_Analysis_NLP.ipynb</code> that you must be able to run without errors to pass the project. If you encounter any issues with the paths, ensure that you follow these requirements in your code:</p>

<ul>
  <li>Whenever you need to run a tokenizer on your sentences, use <code>nltk.tokenize.toktok.ToktokTokenizer</code>.</li>
  <li>When removing stopwords, always use <code>nltk.corpus.stopwords.words('english')</code>.</li>
  <li>For stemming, use <code>nltk.porter.PorterStemmer</code>.</li>
  <li>For lemmatization, use the Spacy pre-trained model <code>en_core_web_sm</code>.</li>
</ul>
--------------------------------------------------------------------------------------------------------

By this, we have come to the end of this project.

Thanks for trying to understand it, I hope you like it,
Mateo.
