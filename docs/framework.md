Frameworks are training wheels. They're most useful when you're learning, and they become less useful once you've internalized them. The senior version of you won't consult a checklist — you'll just know what to do next because you've done it fifty times. What I'm about to give you is scaffolding, not scripture. Use it to build the reflex, then let it fade into the background as intuition.

One more caveat: the checklists differ by problem type more than by phase. A regression project and a classification project share Phase 1 almost entirely but diverge sharply in Phase 2 (what you look for) and Phase 5 (how you evaluate). NLP diverges from both in Phase 2 and Phase 3 (data processing looks very different when your inputs are text). I'll organize by phase, then within each phase call out where problem type matters.

## Phase 1 — Define

# Universal across problem types. You should never leave this phase without a written answer to all four questions below. If any of them is fuzzy, code will not save you.

1 - The problem in one sentence. Must include the input, the output, and the decision the output supports. "Predict Y from X to help someone do Z."
2 - The input and output, with a concrete example. One real row or record. If you can't produce one from actual data, you don't have a project yet.
3 - The success metric. Chosen by problem type — see below. Include a target threshold and a baseline to beat.
4 - The scope decision. What's in v1, what's deferred. What data slice, what time period, what user segment. Every project has a natural tendency to expand; naming what's not in v1 is a discipline.

# Metric choice by problem type — the piece you asked about:

- Regression (predicting a continuous number: price, delay minutes, temperature): default to RMSE or MAE. Use RMSE when large errors matter disproportionately (like flight delays where a 3-hour miss is much worse than a 5-minute one), MAE when you want the error in the same units as the target and don't want outliers to dominate. Add R² as a secondary metric if you want a "how much variance did I explain" story, but don't lead with it.
- Binary classification balanced (roughly 50/50 classes): accuracy is fine, add F1 as backup.
- Binary classification imbalanced (like flight delays, fraud, disease): ROC-AUC (measures ranking quality regardless of threshold), or PR-AUC if the positive class is very rare (<5%). Never lead with accuracy.
- Multi-class classification balanced: accuracy + confusion matrix.
- Multi-class classification imbalanced (like your NOTAM problem): macro-averaged F1 (equal weight per class regardless of size). Micro-F1 is what you'd use if you cared about total right-vs-wrong count.
- Text generation / summarization: no clean automatic metric. Use ROUGE and BLEU as noisy proxies, but plan for human evaluation of 30-100 examples for the real signal.

# Universal rule: 
Always define a baseline. Predict the mean (regression), predict the majority class (classification), predict the previous value (time series), keyword search (retrieval). If your model doesn't beat the baseline meaningfully, you've either got a broken pipeline or an unlearnable problem.

## Phase 2 — Explore

This is the phase where the checklist matters most because it's the phase where what you don't check will hurt you later. Here's the universal core, then problem-type additions.

# Universal exploration (do these for every project):

1- Shape. Rows × columns. How much data do you actually have?
2- Dtypes. Is anything typed weirdly (dates as strings, categories as ints, IDs as floats)?
3- Missingness. Which columns have nulls, how many, and — most importantly — why. Missing data is almost never random; it usually has an operational cause worth understanding.
4- Duplicates. How many exact-duplicate rows? Are duplicates legitimate or a data pipeline bug?
5- Target distribution. What does your target variable actually look like?
6- The leakage audit. For every column, ask: "would I know this value at prediction time?" List forbidden features explicitly.
7- One or two obvious relationships. Pick a feature you suspect matters and check whether reality agrees. This calibrates your intuition against the data.

# Regression-specific additions:

- Target distribution shape. Is it normal? Log-normal? Skewed? Bimodal? Many regression models assume a roughly normal target; heavy skew often means log-transforming the target helps. Check with a histogram.
- Outliers. Regression is far more sensitive to outliers than classification. A single row with a 500-minute delay when everything else is under 100 will pull your model. Decide whether outliers are real or errors, and whether to keep/cap/remove them.
- Feature-target correlation. For numeric features, compute Pearson or Spearman correlation with the target. Rank them. This gives you a sense of which features carry signal before you model.
- Range checks. Prices can't be negative. Ages can't be 300. Distances can't be zero for two different airports. Find impossibilities.

# Classification-specific additions:

- Class balance. What fraction of rows are each class? This drives your metric choice (see Phase 1) and your sampling strategy.
- Class ambiguity. Are any two classes hard to tell apart even for a human expert? If yes, that's your model's error ceiling.
- Base rate per key subgroup. Does class distribution vary a lot across, say, geography or time? A national-level 20% positive rate hiding a per-region range of 5-40% is important to know.

# NLP-specific additions (which apply to your current project):

- Text length distribution. Not just "some are long" — get the distribution: df['text'].str.len().describe() and a histogram. This drives your max_length choice for transformer models. Same for token counts if you have a tokenizer.
- Vocabulary characteristics. Are there many domain-specific abbreviations, acronyms, or technical terms? Are they consistent (RWY always means runway) or ambiguous?
- Language. Is everything in one language, or mixed? Any encoding issues (weird characters, ASCII vs UTF-8 problems)?
- Class-conditional length. Do longer texts belong to certain classes? Sometimes short texts get systematically mis-classified because they lack context.
- Look at 5-10 examples per class. Read them. Not skim — read. This builds domain intuition no summary statistic will give you. In your NOTAM project, this is where you'd notice that ILS-related NOTAMs have a distinct vocabulary from taxiway ones.
- Class definitions review. If the dataset comes with labeling guidelines (like DEEL-AI's), read them. This tells you where your model will legitimately struggle because humans struggle there too.

## Phase 3 — Prototype

# Universal principles:

1- Smallest end-to-end pipeline first. Data loading → minimum features → simplest model → evaluation number. One script. Ugly is fine. The goal is a number, not a good number.
2- The simplest model that runs. Logistic regression for classification. Linear regression for regression. TF-IDF + logistic regression for NLP classification. A prompt to a general-purpose LLM for NLP generation. Never start with the fancy model. The simple model is your baseline; you'll compare everything else against it.
3- The baseline is also a Phase 3 deliverable. Not just the model — the naive comparison too. Same principle across all problem types: predict-the-mean for regression, majority-class for classification, most-frequent-label per group for structured problems.
4- A working train → predict → score loop. If you can produce one honest number end-to-end, you've earned Phase 4.

# Problem-type note: 
For NLP, the "simplest model that runs" often means TF-IDF vectorization + logistic regression. It's 15 lines of scikit-learn code, and it's a surprisingly strong baseline for text classification — often within 5-10 points of a fine-tuned transformer. Any transformer model you build later gets compared against this.

## Phase 4 — Refactor and Iterate

# Universal principles:

1- Move logic out of scripts into modules. Same file layout you already know: data.py, features.py, model.py, train.py, evaluate.py, config.py. Scripts have if __name__ == "__main__": guards; modules are pure.
2- Config-driven everything. Every magic number, every path, every hyperparameter in config.py.
3- Experiment log. One row per training run, mechanically written. Model, features, metric on train, metric on val, baseline metric. Same pattern regardless of problem type.
4- Change one thing per experiment. The rule that separates measurable progress from vibes.
5- Read the train/val gap to detect overfitting. If train metric is much better than val metric, your model is memorizing.
6- A proper validation set. Test is sacred. Val is where you iterate.

# Regression-specific additions:

- Feature scaling matters more (many regression models assume features on similar scales). Standard scaler on numeric features before linear regression.
- Watch for target leakage from features that are functions of the target (e.g., if you have "total sale amount" and are predicting "unit price," the total is the price × quantity — leaky).

# Classification-specific additions:

- If severely imbalanced, consider class weights (class_weight='balanced' in scikit-learn) before trying resampling. Simpler and often equally effective.
- Look at the confusion matrix, not just the aggregate metric. Which classes are getting confused for which? The answer often suggests the next feature to add.

# NLP-specific additions:

- Your "features" are usually a tokenizer plus embeddings. Standard scaler doesn't apply.
- Preprocessing decisions (lowercase? remove punctuation? handle abbreviations?) become hyperparameters worth ablating.
- Track prompt versions in your experiment log if you're using LLMs. A prompt is a hyperparameter.

## Phase 5 — Ship

# Universal, and where problem type matters least. You already know this one.

1- Final evaluation on test set. Once. Report the number.
2- README with problem, approach, findings, setup, limitations, reproducibility.
3- Results notebook or dashboard with the story of your experiments.
4- At least one unit test proving your feature or preprocessing function is testable in isolation.
5- Clean git log that reads as a project narrative.
6- Personal reflection — what did you learn?

# The only problem-type variation: 
regression models get error distribution plots (histogram of residuals, actual vs predicted scatter). Classification models get confusion matrices and per-class F1. NLP models get error analysis on misclassified examples — read 20 wrong predictions and look for patterns.

## Two habits that transcend all phases

I want to name these separately because they're the highest-leverage things you can carry across every project you ever do.

# The leakage audit is continuous. 
In every phase, for every value going into any metric or model: "what computation produced this, and was the information allowed?" Feature leakage in Phase 2, train/val contamination in Phase 3, evaluation leakage in Phase 4. Same question, different disguises.

# Baselines are non-negotiable. 
In every phase, ask: "compared to what?" A raw metric number is meaningless without a baseline. A model at 0.65 AUC is impressive if the baseline is 0.55 and useless if the baseline is 0.72. Same principle for regression, classification, ranking, generation, everything.