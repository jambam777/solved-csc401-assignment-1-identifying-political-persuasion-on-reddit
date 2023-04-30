Download Link: https://assignmentchef.com/product/solved-csc401-assignment-1-identifying-political-persuasion-on-reddit
<br>
<h1>1           Pre-processing, tokenizing, and tagging</h1>

The comments, as given, are not in a form amenable to feature extraction for classification – there is too much ‘noise’. Therefore, the first step is to complete a Python program named a1 preproc.py, in accordance with Section 5, that will read subsets of JSON files, and for each comment perform the following steps, in order, on the ‘body’ field of each selected comment:

<ol>

 <li>Remove all newline characters.</li>

 <li>Replace HTML character codes (i.e., <em>&amp;…;</em>) with their ASCII equivalent (see <a href="http://www.asciitable.com">http://www.asciitable.com</a><a href="http://www.asciitable.com">)</a>.</li>

 <li>Remove all URLs (i.e., tokens beginning with <em>http </em>or <em>www</em>).</li>

 <li>Split each punctuation (see punctuation) into its own token using whitespace except:

  <ul>

   <li></li>

   <li>Periods in abbreviations (e.g., <em>g.</em>) are <em>not </em>split from their tokens. E.g., <em>e.g. </em>stays <em>e.g.</em></li>

   <li>Multiple punctuation (e.g., <em>!?!</em>, <em>…</em>) are <em>not </em>split internally. E.g., <em>Hi!!! </em>becomes <em>Hi !!!</em></li>

   <li>You can handle single hyphens (-) between words as you please. E.g., you can split <em>non-committal </em>into three tokens or leave it as one.</li>

  </ul></li>

 <li>Split clitics using whitespace.

  <ul>

   <li>Clitics are contracted forms of words, such as <em>n’t</em>, that are concatenated with the previous word.</li>

   <li>Note: the possessive <em>’s </em>has its own tag and is distinct from the clitic <em>’s</em>, but nonetheless must be separated by a space; likewise, the possessive on plurals must be separated (e.g., <em>dogs ’</em>).</li>

  </ul></li>

 <li>Each token is tagged with its part-of-speech using spaCy (see below).

  <ul>

   <li>A tagged token consists of a word, the ‘/’ symbol, and the tag (e.g., <em>dog/NN</em>). See below for information on how to use the tagging module. The tagger can make mistakes.</li>

  </ul></li>

 <li>Remove stopwords. See /u/cs401/Wordlists/StopWords.</li>

 <li>Apply lemmatization using spaCy (see below).</li>

 <li>Add a newline between each sentence.

  <ul>

   <li>This will require detecting end-of-sentence punctuation. Some punctuation does not end a sentence; see standard abbreviations here: /u/cs401/Wordlists/abbrev.english. It can be difficult to detect when an abbreviation ends a sentence; e.g., in <em>Go to St. John’s St. John is there.</em>, the first period is used in an abbreviation, the last period ends a sentence, and the second period is used both in an abbreviation and an end-of-sentence. You are not expected to write a ‘perfect’ pre-processor (none exists!), but merely to use your best judgment in writing heuristics; see section 4.2.4 of the Manning and Schu¨tze text for ideas.</li>

  </ul></li>

 <li>Convert text to lowercase.</li>

</ol>

<strong>Functionality: </strong>The a1 preproc.py program reads a subset of the (static) input JSON files, retains

the fields you care about, including ‘<strong>id</strong>’, which you’ll soon use as a key to obtain pre-computed features, and ‘<strong>body</strong>’, which is text that you preprocess and replace before saving the result to an output file. To each comment, also add a <strong>cat </strong>field, with the name of the file from which the comment was retrieved (e.g.,

‘Left’, ‘Alt’,…).

The program takes three arguments: your student ID (mandatory), the output file (mandatory), and the maximum number of lines to sample from each category file (optional; default=10,000). For example, if you are student 999123456 and want to create preproc.json, you’d run:

python a1 preproc.py 999123456 -o preproc.json The output of a1 preproc.py will be used in Task 2.

<strong><u>Your task: </u></strong>Copy the template from /u/cs401/A1/code/a1 preproc.py. There are two functions you need to modify:

<ol>

 <li>In preproc1, fill out each <strong>if </strong>statement with the associated preprocessing step above.</li>

 <li>In main, replace the lines marked with TODO with the code they describe.</li>

</ol>

For this section, you may only use standard Python libraries, <em>except </em>for tagging (step 6) and lemmatization (step 8), as described below. For debugging, you are advised to either use a different input folder with your own JSON data, or pass strings directly to preproc1.

<strong>spaCy: </strong><a href="https://spacy.io">spaCy</a> is a Python library for natural language processing tasks, especially in information

extraction. Here, we <strong>only </strong>use its ability to obtain part-of-speech tags and lemma. For example:

import spacy

nlp = spacy.load(’en’, disable=[’parser’, ’ner’]) utt = nlp(u”I know the best words”) for token in utt:

…                            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,

…                                       token.shape_, token.is_alpha, token.is_stop)

When performing step (6) above, simply append ‘/’ and the appropriate token.tagto the end of each tag, as per the earlier example.

When performing step (8) above, simply replace the token itself with the token.lemma. E.g., <em>words/NNS </em>becomes <em>word/NNS</em>. If the lemma begins with a dash (‘-’) when the token doesn’t (e.g., <em>-PRON- </em>for <em>I</em>, just keep the token.

<strong>Subsampling: </strong>By default, you should only sample 10,000 lines from each of the Left, Center, Right,

and Alt files, for a total of 40,000 lines. From each file, start sampling lines at index [<em>ID </em>% <em>len</em>(<em>X</em>)], where <em>ID </em>is your student ID, % is the modulo arithmetic operator, and <em>len</em>(<em>X</em>) is the number of comments in the given input file (i.e., len(data), once the JSON parse is done). Use circular list indexing if your start index is too close to the ‘end’.

<h1>2           Feature extraction</h1>

The second step is to complete a Python program named a1 extractFeatures.py, in accordance with Section 5, that takes the preprocessed comments from Task 1, extracts features that are relevant to bias detection, and builds an <a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed">npz</a> datafile that will be used to train models and classify comments in Task 3.

For each comment, you need to extract 173 features and write these, along with the category, to a single <a href="https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.array.html">NumPy array.</a> These features are listed below. Several of these features involve counting tokens based on

their tags. For example, counting the number of <em>adverbs </em>in a comment involves counting the number of tokens that have been tagged as RB, RBR, or RBS. Table 4 explicitly defines some of these features; other definitions are available on CDF in /u/cs401/Wordlists/. You may copy and modify these files, but do not change their filenames.

<ol>

 <li>Number of first-person pronouns</li>

 <li>Number of second-person pronouns</li>

 <li>Number of third-person pronouns</li>

 <li>Number of coordinating conjunctions</li>

 <li>Number of past-tense verbs</li>

 <li>Number of future-tense verbs</li>

 <li>Number of commas</li>

 <li>Number of multi-character punctuation tokens</li>

 <li>Number of common nouns</li>

 <li>Number of proper nouns</li>

 <li>Number of adverbs</li>

 <li>Number of <em>wh- </em>words</li>

 <li>Number of slang acronyms</li>

 <li>Number of words in uppercase (≥ 3 letters long)</li>

 <li>Average length of sentences, in tokens</li>

 <li>Average length of tokens, excluding punctuation-only tokens, in characters</li>

 <li>Number of sentences.</li>

 <li>Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms</li>

 <li>Average of IMG from Bristol, Gilhooly, and Logie norms</li>

 <li>Average of FAM from Bristol, Gilhooly, and Logie norms</li>

 <li>Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms</li>

 <li>Standard deviation of IMG from Bristol, Gilhooly, and Logie norms</li>

 <li>Standard deviation of FAM from Bristol, Gilhooly, and Logie norms</li>

 <li>Average of V.Mean.Sum from Warringer norms</li>

 <li>Average of A.Mean.Sum from Warringer norms</li>

 <li>Average of D.Mean.Sum from Warringer norms</li>

 <li>Standard deviation of V.Mean.Sum from Warringer norms</li>

 <li>Standard deviation of A.Mean.Sum from Warringer norms</li>

 <li>Standard deviation of D.Mean.Sum from Warringer norms</li>

</ol>

30-173. LIWC/Receptiv<em>iti </em>features

<strong>Functionality: </strong>The a1 extractFeatures.py program reads a preprocessed JSON file and extracts

features for each comment therein, producing and saving a <em>D</em>×174 NumPy array, where the <em>i<sup>th </sup></em>row is the features for the <em>i<sup>th </sup></em>comment, followed by an integer for the class (0: Left, 1: Center, 2: Right, 3: Alt), as per the <strong>cat </strong>JSON field.

The program takes two arguments: the input filename (i.e., the output of a1 preproc), and the output filename. For example, given input preproc.json and the desired output feats.npz, you’d run:

python a1 extractFeatures.py -i preproc.json -o out.json The output of a1 extractFeatures.py will be used in Task 3.

<strong><u>Your task: </u></strong>Copy the template from /u/cs401/A1/code/a1 extractFeatures.py. There are two functions you need to modify:

<ol>

 <li>In extract1, extract each of the 173 aforementioned features from the input string.</li>

 <li>In main, call extract1 on each datum, and add the results (+ the class) to the feats When your feature extractor works to your satisfaction, build feats.npz, from all input data.</li>

</ol>

<strong><u>Norms: </u></strong>Lexical norms are aggregate subjective scores given to words by a large group of individuals. Each type of norm assigns a numerical value to each word. Here, we use two sets of norms:

<strong>Bristol+GilhoolyLogie: </strong>These are found in /u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv, specifically the fourth, fifth, and sixth columns. These measure the Age-of-acquisition (AoA), imageability (IMG), and familiarity (FAM) of each word, which we can use to measure lexical complexity. More information can be found, for example, <u><a href="https://link.springer.com/article/10.3758/BF03201693">here</a></u><a href="https://link.springer.com/article/10.3758/BF03201693">.</a>

<strong>Warringer: </strong>These are found in /u/cs401/Wordlists/Ratings Warriner et al.csv, specifically the third, sixth, and ninth columns. These norms measure the valence (V), arousal (A), and dominance (D) of each word, according to the <u><a href="https://link.springer.com/article/10.3758/s13428-012-0314-x">VAD</a> </u>model of human affect and emotion. More information on this particular data set can be found <u><a href="http://crr.ugent.be/archives/1003">here</a></u><a href="http://crr.ugent.be/archives/1003">.</a>

When you compute features 18-29, only consider those words that exist in the respective norms file.

<strong>LIWC/Receptiv</strong><em>iti</em><strong>: </strong>The Linguistic Inquiry &amp; Word Count (LIWC) tool has been a standard in

a variety of NLP research, especially around authorship and sentiment analysis. This tool provides 85 measures mostly related to word choice; more information can be found <u><a href="https://liwc.wpengine.com/how-it-works/">here</a></u><a href="https://liwc.wpengine.com/how-it-works/">.</a> The company <a href="https://www.receptiviti.ai">Receptiv</a><em><a href="https://www.receptiviti.ai">iti</a></em>

provides a superset of these features, which also includes 59 measures of personality derived from text. The company has graciously donated access to its API for the purposes of this course.

To simplify things, we have already extracted these 144 features for you. Simply copy the pre-computed features from the appropriate <em>uncompressed </em>npy files stored in /u/cs401/A1/feats/. Specifically:

<ol>

 <li>Comment IDs are stored in txt files (e.g., Alt IDs.txt). When processing a comment, find the index (row) <em>i </em>of the ID in the appropriate ID text file, for the category, and copy the 144 elements, starting at element <em>i </em>· 144, from the associated feats.dat.npy file.</li>

 <li>The file txt provides the names of these features, in the order provided. For this assignment, these names will suffice as to their meaning, but you are welcome to obtain your own API license from <a href="https://www.receptiviti.ai">Receptiv</a><em><a href="https://www.receptiviti.ai">iti</a> </em>in order to get access to their documentation.</li>

</ol>

<h1>3           Experiments and classification</h1>

The third step is to use the features extracted in Task 2 to classify comments using the <u><a href="http://scikit-learn.org/stable/">scikit-learn</a> </u>machine learning package. Here, you will modify various hyper-parameters and interpret the results analytically. As everyone has different slices of the data, there are no expectations on overall <em>accuracy</em>, but you are expected to discuss your findings with scientific rigour. Copy the template from /u/cs401/A1/code/a1 classify.py and complete the main body and the functions for the following experiments according to the specifications therein:

<h2>3.1   Classifiers</h2>

Use the <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split">train</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split">test</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split">split</a> method to split the data into a random 80% for training and 20% for testing.

Train the following 5 classifiers (see hyperlinks for API) with fit(X train, y train):

<ol>

 <li><u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">SVC</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">:</a> support vector machine with a linear kernel.</li>

 <li><u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">SVC</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">:</a> support vector machine with a radial basis function (<em>γ </em>= 2) kernel.</li>

 <li><u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier">RandomForestClassifier</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier">:</a> with a maximum depth of 5, and 10 estimators.</li>

 <li><u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier">MLPClassifier</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier">:</a> A feed-forward neural network, with <em>α </em>= 0<em>.</em></li>

 <li><u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier">AdaBoostClassifier</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier">:</a> with the default hyper-parameters.</li>

</ol>

Here, X train is the first 173 columns of your training data, and y train is the last column. Obtain predicted labels with these classifiers using predict(X test), where X test is the first 173 columns of your testing data. Obtain the 4 × 4 confusion matrix <em>C </em>using <u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">confusion</a> </u><u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">matrix</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">.</a> Given that the element at row <em>i</em>, column <em>j </em>in <em>C </em>(i.e., <em>c<sub>i,j</sub></em>) is the number of instances belonging to class <em>i </em>that were classified as class <em>j</em>, compute the following manually, using the associated function templates:

<strong>Accuracy </strong>: the total number of correctly classified instances over all classifications: .

<strong>Recall </strong>: <em>for each class κ</em>, the fraction of cases that are truly class <em>κ </em>that were classified as.

<strong>Precision </strong>: <em>for each class κ</em>, the fraction of cases classified as <em>κ </em>that truly are .

Write the results to the text file a1 3.1.csv, in a comma-separated value format. Each of the first five lines has the following, in order:

<ol>

 <li>the number of the classifier (i.e., 1-5)</li>

 <li>the overall accuracy, recall for the 4 classes, and precision for the 4 classes</li>

 <li>the confusion matrix, read row-by-row</li>

</ol>

That is, each of the first five lines should have 1+(1+4+4)+4×4 = 26 numbers separated by commas. If so desired, add any analytical commentary to the sixth line of this file.

<h2>3.2   Amount of training data</h2>

Many researchers attribute the success of modern machine learning to the sheer volume of data that is now available. Modify the amount of data that is used to train your preferred classifier from above in five increments: 1K, 5K, 10K, 15K, and 20K. These can be sampled arbitrarily from the training set in Section 3.1. <em>Using only the classification algorithm with the highest accuracy from Section 3.1</em>, report the resulting <em>accuracies </em>in a comma-separated form in the first line of a1 3.2.csv. On the second line of that file, comment on the changes to accuracy as the number of training samples increases, including at least two sentences on a possible explanation. Is there an expected trend? Do you see such a trend? Hypothesize as to why or why not.

<h2>3.3   Feature analysis</h2>

Certain features may be more or less useful for classification, and too many can lead to overfitting or other problems. Here, you will select the best features for classification using <u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest">SelectKBest</a> </u>according to the <a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif">f</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif">classif</a> metric as in:

from sklearn.feature_selection import SelectKBest from sklearn.feature_selection import f_classif

selector = SelectKBest(f_classif, you_figure_it_out) X_new = selector.fit_transform(X_train, y_train) pp = selector.pvalues_

In the example above, pp stores the <em>p</em>-value associated with doing a <em>χ</em><sup>2 </sup>statistical test on each feature. A smaller value means the associated feature better separates the classes. Do this:

<ol>

 <li>For each of the 1K training set from Section 3.2, and the original 32K training set from Section3.1, and for each number of features <em>k </em>= {5<em>,</em>10<em>,</em>20<em>,</em>30<em>,</em>40<em>,</em>50}, find the best <em>k </em>features according to this approach. On each of the first 6 lines of a1 3.3.csv, write the number of features, and the associated <em>p</em>-values for the 32K training set case, separated by commas.</li>

 <li>Train the best classifier from section 3.1 for each of the 1K training set and the 32K training set,using only the best <em>k </em>= 5 features. On the 7<em><sup>th </sup></em>line of a1 3.3.csv, write the accuracy for the 1K training case and the 32K training case, separated by a comma</li>

 <li>On lines 8 to 10 of a1 3.3.csv, answer the following questions:

  <ul>

   <li>What features, if any, are chosen at both the low and high(er) amounts of input data? Alsoprovide a possible explanation as to why this might be.</li>

   <li>Are <em>p</em>-values generally higher or lower given more or less data? Why or why not?</li>

   <li>Name the top 5 features chosen for the 32K training case. Hypothesize as to why those particularfeatures might differentiate the classes.</li>

  </ul></li>

</ol>

<h2>3.4   Cross-validation</h2>

Many papers in machine learning stick with a single subset of data for training and another for testing (occasionally with a third for validation). This may not be the most honest approach. Is the best classifier from Section 3.1 <em>really </em>the best? For each of the classifiers in Section 3.1, run 5-fold cross-validation given all the initially available data. Specifically, use <u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html">KFold</a></u><a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html">.</a> <strong>Set the shuffle argument to true</strong>.

For each fold, obtain accuracy on the test partition after training on the rest for each classifier, in the original order, and report these on their own line (one per fold) in a1 3.4.csv, separated by commas. Next, compare the accuracies of your best classifier, across the 5 folds, with each of the other 4 classifiers to check if it is <em>significantly </em>better than any others. I.e., given vectors a and b, one for each classifier, containing the accuracy values for each of the respective 5 folds, obtain the <em>p</em>-value from the output S, below:

from scipy import stats

S = stats.ttest_rel(a, b)

You should have 4 <em>p</em>-values. Report each in a1 3.4.csv, on the sixth line, separated by commas, in the order the classifiers appear in Section 3.1. On the seventh line, comment on any significance you observe, or any lack thereof, and hypothesize as to why, in one to three sentences.

<h1>4           Bonus</h1>

We will give up to 15 bonus marks for innovative work going substantially beyond the minimal requirements. These marks can make up for marks lost in other sections of the assignment, but your overall mark for this assignment cannot exceed 100%. The obtainable bonus marks will depend on the complexity of the undertaking, and are at the discretion of the marker. Importantly, your bonus work should not affect our ability to mark the main body of the assignment in any way.

You may decide to pursue any number of tasks of your own design related to this assignment, although you should consult with the instructor or the TA before embarking on such exploration. Certainly, the rest of the assignment takes higher priority. Some ideas:

<ul>

 <li><strong>Identify words that the PoS tagger tags incorrectly </strong>and add code that fixes those mistakes. Does this code introduce new errors elsewhere? E.g., if you always tag <em>dog </em>as a noun to correct a mistake, you will encounter errors when <em>dog </em>should be a verb. How can you mitigate such errors?</li>

 <li>Explore <strong>alternative features </strong>to those extracted in Task 2. What other kinds of variables would be useful in distinguishing affect? Consider, for example, the <a href="https://nlp.stanford.edu/sentiment/">Stanford Deep Learning for Sentiment </a> Test your features empirically as you did in Task 3 and discuss your findings.</li>

 <li>Explore <strong>alternative classification methods </strong>to those used in Task 3. Explore different hyperparameters. Which hyper-parameters give the best empirical performance, and why?</li>

 <li>Learn about <strong>topic modelling </strong>as in <u><a href="https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation">latent Dirichlet allocation</a></u><a href="https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation">.</a> Are there topics that have an effect on the accuracy of the system? E.g., is it easier to tell how someone feels about politicians or about events? People or companies? As there may be class imbalances in the groups, how would you go about evaluating this? Go about evaluating this.</li>

</ul>